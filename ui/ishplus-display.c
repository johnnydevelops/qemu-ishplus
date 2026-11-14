/*
 * iSH+: the VM's display and input for the app, which runs QEMU in its own
 * process (-display none). The app attaches to the first graphic console
 * (virtio-gpu) with callbacks for frames and the cursor, and sends pointer
 * and keyboard input (virtio-tablet, virtio-keyboard) here.
 *
 * Callbacks run on QEMU's main loop thread, with the BQL held; the surface's
 * pixels stay valid until the next switch callback. The input functions take
 * the BQL themselves and can be called from any thread.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/main-loop.h"
#include "ui/console.h"
#include "ui/input.h"
#include "ui/surface.h"
#ifdef CONFIG_OPENGL
#include "ui/egl-helpers.h"
#endif

typedef struct IshplusDisplayCallbacks {
    void *context;
    /* A new surface: x8r8g8b8 pixels (BGRA in memory), or NULL when the
     * console has none. */
    void (*surface)(void *context, const void *pixels, int width, int height, int stride);
    void (*update)(void *context, int x, int y, int width, int height);
    /* A cursor image (a8r8g8b8, not premultiplied), or NULL to hide it. */
    void (*cursor)(void *context, const uint32_t *pixels, int width, int height, int hot_x, int hot_y);
    void (*cursor_position)(void *context, int x, int y, int visible);
} IshplusDisplayCallbacks;

int ishplus_display_attach(const IshplusDisplayCallbacks *callbacks);
void ishplus_display_detach(void);
void ishplus_display_set_size(int width, int height);
void ishplus_input_pointer(int x, int y);
void ishplus_input_button(int button, int down);
void ishplus_input_scroll(int dx, int dy);
void ishplus_input_set_relative(int relative);
void ishplus_input_motion(int dx, int dy);
void ishplus_input_key(int usb_usage, int down);
void ishplus_gpu_pause(int pause);

/*
 * GPU frames (with GPU acceleration, egl-headless): the guest's screen as
 * IOSurfaces (BGRA, first row at the top), shown as they are, instead of a
 * copy in the surface (surface and update callbacks then stop until the
 * screen is back to software, which a NULL surface tells). Called on QEMU's
 * main loop thread; QEMU draws into the IOSurfaces that are not in use, so
 * hold one with IOSurfaceIncrementUseCount while showing it.
 */
typedef void (*IshplusGpuFrame)(void *context, void *iosurface, int width, int height);
void ishplus_display_set_gpu_frames(void *context, IshplusGpuFrame frame);
bool ishplus_gpu_frames_wanted(void);
void ishplus_gpu_frame(void *iosurface, int width, int height);
/* For egl-headless: 0 never attached, 1 attached, 2 detached since; and a
 * number that changes when the app (re)attaches or asks for GPU frames, so
 * it gets the current screen */
int ishplus_display_state(void);
unsigned int ishplus_display_generation(void);

static IshplusDisplayCallbacks callbacks;
static DisplayChangeListener listener;
static bool attached;
static IshplusGpuFrame gpu_frame_callback;
static bool ever_attached;
static unsigned int generation;
static void *gpu_frame_context;

static void lock(bool *locked)
{
    *locked = bql_locked();
    if (!*locked) {
        bql_lock();
    }
}

static void unlock(bool locked)
{
    if (!locked) {
        bql_unlock();
    }
}

static void report_surface(DisplaySurface *surface)
{
    if (!callbacks.surface) {
        return;
    }
    if (!surface || (surface_format(surface) != PIXMAN_x8r8g8b8 &&
        surface_format(surface) != PIXMAN_a8r8g8b8)) {
        callbacks.surface(callbacks.context, NULL, 0, 0, 0);
        return;
    }
    callbacks.surface(callbacks.context, surface_data(surface), surface_width(surface),
                      surface_height(surface), surface_stride(surface));
}

static void display_refresh(DisplayChangeListener *dcl)
{
    graphic_hw_update(dcl->con);
}

static void display_update(DisplayChangeListener *dcl, int x, int y, int w, int h)
{
    if (callbacks.update) {
        callbacks.update(callbacks.context, x, y, w, h);
    }
}

static void display_switch(DisplayChangeListener *dcl, DisplaySurface *surface)
{
    report_surface(surface);
}

static bool display_check_format(DisplayChangeListener *dcl, pixman_format_code_t format)
{
    /* Other formats are converted by the console to its default (x8r8g8b8) */
    return format == PIXMAN_x8r8g8b8 || format == PIXMAN_a8r8g8b8;
}

static void display_mouse_set(DisplayChangeListener *dcl, int x, int y, bool on)
{
    if (callbacks.cursor_position) {
        callbacks.cursor_position(callbacks.context, x, y, on);
    }
}

static void display_cursor_define(DisplayChangeListener *dcl, QEMUCursor *cursor)
{
    if (callbacks.cursor) {
        if (cursor) {
            callbacks.cursor(callbacks.context, cursor->data, cursor->width, cursor->height,
                             cursor->hot_x, cursor->hot_y);
        } else {
            callbacks.cursor(callbacks.context, NULL, 0, 0, 0, 0);
        }
    }
}

static const DisplayChangeListenerOps ops = {
    .dpy_name = "ishplus",
    .dpy_refresh = display_refresh,
    .dpy_gfx_update = display_update,
    .dpy_gfx_switch = display_switch,
    .dpy_gfx_check_format = display_check_format,
    .dpy_mouse_set = display_mouse_set,
    .dpy_cursor_define = display_cursor_define,
};

static QemuConsole *graphic_console(void)
{
    QemuConsole *con;

    for (unsigned int i = 0; (con = qemu_console_lookup_by_index(i)); i++) {
        if (qemu_console_is_graphic(con)) {
            return con;
        }
    }
    return NULL;
}

int ishplus_display_attach(const IshplusDisplayCallbacks *new_callbacks)
{
    bool locked;
    QemuConsole *con;
    int ret = -1;

    lock(&locked);
    if (attached) {
        unregister_displaychangelistener(&listener);
        attached = false;
    }
    con = graphic_console();
    if (con) {
        callbacks = *new_callbacks;
        memset(&listener, 0, sizeof(listener));
        listener.ops = &ops;
        listener.con = con;
        /* Reports the current surface (dpy_gfx_switch) */
        register_displaychangelistener(&listener);
        attached = true;
        ever_attached = true;
        generation++;
        ret = 0;
    }
    unlock(locked);
    return ret;
}

void ishplus_display_detach(void)
{
    bool locked;

    lock(&locked);
    if (attached) {
        unregister_displaychangelistener(&listener);
        attached = false;
    }
    memset(&callbacks, 0, sizeof(callbacks));
    gpu_frame_callback = NULL;
    gpu_frame_context = NULL;
    unlock(locked);
}

/* After ishplus_display_attach; NULL stops them (until the next attach) */
void ishplus_display_set_gpu_frames(void *context, IshplusGpuFrame frame)
{
    bool locked;

    lock(&locked);
    gpu_frame_context = context;
    gpu_frame_callback = frame;
    generation++;
    unlock(locked);
}

int ishplus_display_state(void)
{
    return attached ? 1 : ever_attached ? 2 : 0;
}

unsigned int ishplus_display_generation(void)
{
    return generation;
}

bool ishplus_gpu_frames_wanted(void)
{
    return attached && gpu_frame_callback;
}

void ishplus_gpu_frame(void *iosurface, int width, int height)
{
    if (attached && gpu_frame_callback) {
        gpu_frame_callback(gpu_frame_context, iosurface, width, height);
    }
}

/* The size the guest should use: virtio-gpu offers it as its preferred mode. */
void ishplus_display_set_size(int width, int height)
{
    bool locked;
    QemuConsole *con;

    if (width <= 0 || height <= 0) {
        return;
    }
    lock(&locked);
    con = attached ? listener.con : graphic_console();
    if (con && dpy_ui_info_supported(con)) {
        QemuUIInfo info = *dpy_get_ui_info(con);
        info.width = width;
        info.height = height;
        /* About 110 dpi, so the guest's DPI is ordinary */
        info.width_mm = width * 254 / 1100;
        info.height_mm = height * 254 / 1100;
        dpy_set_ui_info(con, &info, false);
    }
    unlock(locked);
}

static QemuConsole *input_console(void)
{
    return attached ? listener.con : NULL;
}

static int surface_size(QemuConsole *con, bool height)
{
    DisplaySurface *surface = qemu_console_surface(con);

    if (!surface) {
        return 1;
    }
    return height ? surface_height(surface) : surface_width(surface);
}

/* In the surface's pixels */
void ishplus_input_pointer(int x, int y)
{
    bool locked;
    QemuConsole *con;

    lock(&locked);
    con = input_console();
    if (con) {
        qemu_input_queue_abs(con, INPUT_AXIS_X, x, 0, surface_size(con, false));
        qemu_input_queue_abs(con, INPUT_AXIS_Y, y, 0, surface_size(con, true));
        qemu_input_event_sync();
    }
    unlock(locked);
}

/*
 * A captured mouse: relative motion, in the guest's pixels, to the relative
 * mouse (virtio-mouse), which then also gets the buttons and wheel; off,
 * the absolute tablet (ishplus_input_pointer) gets them again.
 */
void ishplus_input_set_relative(int relative)
{
    bool locked;

    lock(&locked);
    qemu_input_handler_activate_mask(relative ? INPUT_EVENT_MASK_REL : INPUT_EVENT_MASK_ABS);
    unlock(locked);
}

void ishplus_input_motion(int dx, int dy)
{
    bool locked;
    QemuConsole *con;

    lock(&locked);
    con = input_console();
    if (con && (dx || dy)) {
        qemu_input_queue_rel(con, INPUT_AXIS_X, dx);
        qemu_input_queue_rel(con, INPUT_AXIS_Y, dy);
        qemu_input_event_sync();
    }
    unlock(locked);
}

/* 0 left, 1 middle, 2 right */
void ishplus_input_button(int button, int down)
{
    static const InputButton buttons[] = {
        INPUT_BUTTON_LEFT, INPUT_BUTTON_MIDDLE, INPUT_BUTTON_RIGHT,
    };
    bool locked;
    QemuConsole *con;

    if (button < 0 || button > 2) {
        return;
    }
    lock(&locked);
    con = input_console();
    if (con) {
        qemu_input_queue_btn(con, buttons[button], down);
        qemu_input_event_sync();
    }
    unlock(locked);
}

/* Wheel clicks: positive dy scrolls down, positive dx right */
void ishplus_input_scroll(int dx, int dy)
{
    bool locked;
    QemuConsole *con;

    lock(&locked);
    con = input_console();
    if (con) {
        for (int i = 0; i < abs(dy); i++) {
            InputButton b = dy > 0 ? INPUT_BUTTON_WHEEL_DOWN : INPUT_BUTTON_WHEEL_UP;
            qemu_input_queue_btn(con, b, true);
            qemu_input_event_sync();
            qemu_input_queue_btn(con, b, false);
            qemu_input_event_sync();
        }
        for (int i = 0; i < abs(dx); i++) {
            InputButton b = dx > 0 ? INPUT_BUTTON_WHEEL_RIGHT : INPUT_BUTTON_WHEEL_LEFT;
            qemu_input_queue_btn(con, b, true);
            qemu_input_event_sync();
            qemu_input_queue_btn(con, b, false);
            qemu_input_event_sync();
        }
    }
    unlock(locked);
}

/* A USB HID keyboard usage (UIKey.keyCode on iOS) */
void ishplus_input_key(int usb_usage, int down)
{
    bool locked;
    QemuConsole *con;
    int qcode;

    if (usb_usage <= 0 || usb_usage >= qemu_input_map_usb_to_qcode_len) {
        return;
    }
    qcode = qemu_input_map_usb_to_qcode[usb_usage];
    if (qcode == Q_KEY_CODE_UNMAPPED) {
        return;
    }
    lock(&locked);
    con = input_console();
    if (con) {
        qemu_input_event_send_key_qcode(con, qcode, down);
    }
    unlock(locked);
}

/*
 * iOS refuses GPU work from apps in the background (Metal command buffers
 * fail). While paused, virtio-gpu leaves the guest's commands queued (the
 * guest waits for them), and the GPU work already submitted is finished
 * before this returns. Call before the app enters the background, and with
 * 0 when it is back.
 */
void ishplus_gpu_pause(int pause)
{
    static bool paused;
    bool locked;
    QemuConsole *con;

    lock(&locked);
    con = attached ? listener.con : graphic_console();
    if (con && pause && !paused) {
        graphic_hw_gl_block(con, true);
        paused = true;
#ifdef CONFIG_OPENGL
        if (qemu_egl_display && qemu_egl_rn_ctx) {
            EGLContext current = eglGetCurrentContext();
            /* Switching contexts flushes the other contexts' work to the
             * one queue that glFinish then waits for */
            eglMakeCurrent(qemu_egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, qemu_egl_rn_ctx);
            glFinish();
            if (current != EGL_NO_CONTEXT) {
                eglMakeCurrent(qemu_egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, current);
            }
        }
#endif
    } else if (con && !pause && paused) {
        paused = false;
        graphic_hw_gl_block(con, false);
    }
    unlock(locked);
}
