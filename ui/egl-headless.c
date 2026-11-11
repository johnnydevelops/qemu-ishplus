#include "qemu/osdep.h"
#include "qemu/error-report.h"
#include "qemu/module.h"
#include "qapi/error.h"
#include "ui/console.h"
#include "ui/egl-helpers.h"
#include "ui/egl-context.h"
#include "ui/shader.h"

#ifdef __APPLE__
/*
 * iSH+: with the app attached for GPU frames (ishplus-display.c), the
 * scanout is blitted on the GPU into IOSurfaces, which the app shows as
 * they are: no readback into the console's surface.
 */
#include <IOSurface/IOSurfaceRef.h>
#include <CoreFoundation/CoreFoundation.h>

bool ishplus_gpu_frames_wanted(void);
void ishplus_gpu_frame(void *iosurface, int width, int height);
int ishplus_display_state(void);
unsigned int ishplus_display_generation(void);

/* virglrenderer's first context (its ctx 0), where the framebuffers are */
static QEMUGLContext first_context;

#ifndef EGL_IOSURFACE_ANGLE
#define EGL_IOSURFACE_ANGLE 0x3454
#define EGL_IOSURFACE_PLANE_ANGLE 0x345A
#define EGL_TEXTURE_RECTANGLE_ANGLE 0x345B
#define EGL_TEXTURE_TYPE_ANGLE 0x345C
#define EGL_TEXTURE_INTERNAL_FORMAT_ANGLE 0x345D
#endif

/* The app holds a buffer while it shows it (IOSurfaceIncrementUseCount) */
#define GPU_BUFFERS 4

typedef struct gpu_buffer {
    IOSurfaceRef surface;
    EGLSurface pbuffer;
    egl_fb fb;
    uint64_t used;    /* when it last got a frame */
} gpu_buffer;
#endif

typedef struct egl_dpy {
    DisplayChangeListener dcl;
    DisplaySurface *ds;
    QemuGLShader *gls;
    egl_fb guest_fb;
    egl_fb cursor_fb;
    egl_fb blit_fb;
    bool y_0_top;
    uint32_t pos_x;
    uint32_t pos_y;
#ifdef __APPLE__
    gpu_buffer gpu[GPU_BUFFERS];
    uint64_t gpu_frames;
    bool gpu_active;
    unsigned int generation;    /* the app's, when it last got a frame */
#endif
} egl_dpy;

#ifdef __APPLE__
static void gpu_buffer_destroy(gpu_buffer *buf)
{
    if (buf->pbuffer != EGL_NO_SURFACE) {
        egl_fb_destroy(&buf->fb);    /* deletes the texture too */
        eglReleaseTexImage(qemu_egl_display, buf->pbuffer, EGL_BACK_BUFFER);
        eglDestroySurface(qemu_egl_display, buf->pbuffer);
    }
    if (buf->surface) {
        CFRelease(buf->surface);
    }
    memset(buf, 0, sizeof(*buf));
    buf->pbuffer = EGL_NO_SURFACE;
}

static void gpu_buffers_destroy(egl_dpy *edpy)
{
    for (int i = 0; i < GPU_BUFFERS; i++) {
        gpu_buffer_destroy(&edpy->gpu[i]);
    }
}

static CFNumberRef number(int value)
{
    return CFNumberCreate(NULL, kCFNumberIntType, &value);
}

static bool gpu_buffer_create(gpu_buffer *buf, int width, int height)
{
    static EGLConfig config;
    const void *keys[] = {
        kIOSurfaceWidth, kIOSurfaceHeight, kIOSurfaceBytesPerElement,
        kIOSurfacePixelFormat,
    };
    const void *values[] = {
        number(width), number(height), number(4), number('BGRA'),
    };
    CFDictionaryRef props;
    GLuint texture;

    if (!config) {
        static const EGLint want[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_BIND_TO_TEXTURE_RGBA, EGL_TRUE,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
            EGL_NONE,
        };
        EGLint n = 0;
        if (!eglChooseConfig(qemu_egl_display, want, &config, 1, &n) || n < 1) {
            config = NULL;
            return false;
        }
    }

    props = CFDictionaryCreate(NULL, keys, values, 4, &kCFTypeDictionaryKeyCallBacks,
                               &kCFTypeDictionaryValueCallBacks);
    for (int i = 0; i < 4; i++) {
        CFRelease(values[i]);
    }
    buf->surface = IOSurfaceCreate(props);
    CFRelease(props);
    if (!buf->surface) {
        return false;
    }

    const EGLint attribs[] = {
        EGL_WIDTH, width, EGL_HEIGHT, height,
        EGL_IOSURFACE_PLANE_ANGLE, 0,
        EGL_TEXTURE_TARGET, EGL_TEXTURE_2D,
        EGL_TEXTURE_INTERNAL_FORMAT_ANGLE, GL_BGRA_EXT,
        EGL_TEXTURE_FORMAT, EGL_TEXTURE_RGBA,
        EGL_TEXTURE_TYPE_ANGLE, GL_UNSIGNED_BYTE,
        EGL_NONE,
    };
    buf->pbuffer = eglCreatePbufferFromClientBuffer(qemu_egl_display, EGL_IOSURFACE_ANGLE,
                                                    buf->surface, config, attribs);
    if (buf->pbuffer == EGL_NO_SURFACE) {
        error_report("egl-headless: no IOSurface pbuffer (0x%x)", eglGetError());
        gpu_buffer_destroy(buf);
        return false;
    }
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    if (!eglBindTexImage(qemu_egl_display, buf->pbuffer, EGL_BACK_BUFFER)) {
        error_report("egl-headless: IOSurface not bound (0x%x)", eglGetError());
        glDeleteTextures(1, &texture);
        gpu_buffer_destroy(buf);
        return false;
    }
    egl_fb_setup_for_tex(&buf->fb, width, height, texture, true);
    return true;
}

/* A buffer the app is not showing (or the least recent), of this size */
static gpu_buffer *gpu_buffer_next(egl_dpy *edpy, int width, int height)
{
    gpu_buffer *best = NULL;

    for (int i = 0; i < GPU_BUFFERS; i++) {
        gpu_buffer *buf = &edpy->gpu[i];
        if (buf->surface && (buf->fb.width != width || buf->fb.height != height)) {
            gpu_buffer_destroy(buf);
        }
    }
    for (int i = 0; i < GPU_BUFFERS; i++) {
        gpu_buffer *buf = &edpy->gpu[i];
        if (buf->surface && !IOSurfaceIsInUse(buf->surface) &&
            (!best || buf->used < best->used)) {
            best = buf;
        }
    }
    if (best) {
        return best;
    }
    for (int i = 0; i < GPU_BUFFERS; i++) {
        if (!edpy->gpu[i].surface) {
            return gpu_buffer_create(&edpy->gpu[i], width, height) ? &edpy->gpu[i] : NULL;
        }
    }
    /* All shown: the least recent */
    for (int i = 0; i < GPU_BUFFERS; i++) {
        if (!best || edpy->gpu[i].used < best->used) {
            best = &edpy->gpu[i];
        }
    }
    return best;
}

/* The scanout to the app, on the GPU; false to read it back instead */
static bool gpu_frame(egl_dpy *edpy)
{
    gpu_buffer *buf;

    if (!ishplus_gpu_frames_wanted()) {
        if (edpy->gpu_active) {
            edpy->gpu_active = false;
            gpu_buffers_destroy(edpy);
        }
        return false;
    }
    buf = gpu_buffer_next(edpy, edpy->guest_fb.width, edpy->guest_fb.height);
    if (!buf) {
        return false;
    }
    /* As for the readback below: the first row at y 0, the IOSurface's
     * first row */
    if (edpy->cursor_fb.texture) {
        egl_texture_blit(edpy->gls, &buf->fb, &edpy->guest_fb, !edpy->y_0_top);
        egl_texture_blend(edpy->gls, &buf->fb, &edpy->cursor_fb,
                          !edpy->y_0_top, edpy->pos_x, edpy->pos_y, 1.0, 1.0);
    } else {
        egl_fb_blit(&buf->fb, &edpy->guest_fb, edpy->y_0_top);
    }
    /* Done on the GPU before the app shows it */
    glFinish();
    buf->used = ++edpy->gpu_frames;
    edpy->gpu_active = true;
    ishplus_gpu_frame(buf->surface, buf->fb.width, buf->fb.height);
    return true;
}
#endif

/* ------------------------------------------------------------------ */

static void egl_scanout_flush(DisplayChangeListener *dcl,
                              uint32_t x, uint32_t y,
                              uint32_t w, uint32_t h);

static void egl_refresh(DisplayChangeListener *dcl)
{
#ifdef __APPLE__
    egl_dpy *edpy = container_of(dcl, egl_dpy, dcl);

    /* The app attached again, or asked for GPU frames: the screen it has
     * may be old, and an idle guest sends no new one */
    if (ishplus_display_state() == 1 && edpy->generation != ishplus_display_generation() &&
        edpy->guest_fb.texture && edpy->ds && first_context &&
        !qemu_console_is_gl_blocked(dcl->con)) {
        EGLContext current = eglGetCurrentContext();
        eglMakeCurrent(qemu_egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, first_context);
        egl_scanout_flush(dcl, 0, 0, edpy->guest_fb.width, edpy->guest_fb.height);
        eglMakeCurrent(qemu_egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, current);
    }
#endif
    graphic_hw_update(dcl->con);
}

static void egl_gfx_update(DisplayChangeListener *dcl,
                           int x, int y, int w, int h)
{
}

static void egl_gfx_switch(DisplayChangeListener *dcl,
                           struct DisplaySurface *new_surface)
{
    egl_dpy *edpy = container_of(dcl, egl_dpy, dcl);

    edpy->ds = new_surface;
}

static QEMUGLContext egl_create_context(DisplayGLCtx *dgc,
                                        QEMUGLParams *params)
{
    QEMUGLContext ctx;

    eglMakeCurrent(qemu_egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE,
                   qemu_egl_rn_ctx);
    ctx = qemu_egl_create_context(dgc, params);
#ifdef __APPLE__
    if (!first_context) {
        first_context = ctx;
    }
#endif
    return ctx;
}

static void egl_destroy_context(DisplayGLCtx *dgc, QEMUGLContext ctx)
{
#ifdef __APPLE__
    if (ctx == first_context) {
        first_context = NULL;
    }
#endif
    qemu_egl_destroy_context(dgc, ctx);
}

static void egl_scanout_disable(DisplayChangeListener *dcl)
{
    egl_dpy *edpy = container_of(dcl, egl_dpy, dcl);

    egl_fb_destroy(&edpy->guest_fb);
    egl_fb_destroy(&edpy->blit_fb);
#ifdef __APPLE__
    if (edpy->gpu_active) {
        edpy->gpu_active = false;
        gpu_buffers_destroy(edpy);
        ishplus_gpu_frame(NULL, 0, 0);
    }
#endif
}

static void egl_scanout_texture(DisplayChangeListener *dcl,
                                uint32_t backing_id,
                                bool backing_y_0_top,
                                uint32_t backing_width,
                                uint32_t backing_height,
                                uint32_t x, uint32_t y,
                                uint32_t w, uint32_t h,
                                ScanoutTextureNative native)
{
    egl_dpy *edpy = container_of(dcl, egl_dpy, dcl);

    edpy->y_0_top = backing_y_0_top;

    /* source framebuffer */
    egl_fb_setup_for_tex(&edpy->guest_fb,
                         backing_width, backing_height, backing_id, false);

    /* dest framebuffer */
    if (edpy->blit_fb.width  != backing_width ||
        edpy->blit_fb.height != backing_height) {
        egl_fb_destroy(&edpy->blit_fb);
        egl_fb_setup_new_tex(&edpy->blit_fb, backing_width, backing_height);
    }
}

#ifdef CONFIG_GBM

static void egl_scanout_dmabuf(DisplayChangeListener *dcl,
                               QemuDmaBuf *dmabuf)
{
    uint32_t width, height, texture;

    egl_dmabuf_import_texture(dmabuf);
    texture = qemu_dmabuf_get_texture(dmabuf);
    if (!texture) {
        return;
    }

    width = qemu_dmabuf_get_width(dmabuf);
    height = qemu_dmabuf_get_height(dmabuf);

    egl_scanout_texture(dcl, texture, false, width, height, 0, 0,
                        width, height, NO_NATIVE_TEXTURE);
}

static void egl_cursor_dmabuf(DisplayChangeListener *dcl,
                              QemuDmaBuf *dmabuf, bool have_hot,
                              uint32_t hot_x, uint32_t hot_y)
{
    uint32_t width, height, texture;
    egl_dpy *edpy = container_of(dcl, egl_dpy, dcl);

    if (dmabuf) {
        egl_dmabuf_import_texture(dmabuf);
        texture = qemu_dmabuf_get_texture(dmabuf);
        if (!texture) {
            return;
        }

        width = qemu_dmabuf_get_width(dmabuf);
        height = qemu_dmabuf_get_height(dmabuf);
        egl_fb_setup_for_tex(&edpy->cursor_fb, width, height, texture, false);
    } else {
        egl_fb_destroy(&edpy->cursor_fb);
    }
}

static void egl_release_dmabuf(DisplayChangeListener *dcl,
                               QemuDmaBuf *dmabuf)
{
    egl_dmabuf_release_texture(dmabuf);
}

#endif

static void egl_cursor_position(DisplayChangeListener *dcl,
                                uint32_t pos_x, uint32_t pos_y)
{
    egl_dpy *edpy = container_of(dcl, egl_dpy, dcl);

    edpy->pos_x = pos_x;
    edpy->pos_y = pos_y;
}

static void egl_scanout_flush(DisplayChangeListener *dcl,
                              uint32_t x, uint32_t y,
                              uint32_t w, uint32_t h)
{
    egl_dpy *edpy = container_of(dcl, egl_dpy, dcl);

    if (!edpy->guest_fb.texture || !edpy->ds) {
        return;
    }
    assert(surface_format(edpy->ds) == PIXMAN_x8r8g8b8);

#ifdef __APPLE__
    /* The app is away: no one to show it to (it gets the screen when it
     * is back, see egl_refresh) */
    if (ishplus_display_state() == 2) {
        return;
    }
    edpy->generation = ishplus_display_generation();
    if (gpu_frame(edpy)) {
        return;
    }
#endif

    if (edpy->cursor_fb.texture) {
        /* have cursor -> render using textures */
        egl_texture_blit(edpy->gls, &edpy->blit_fb, &edpy->guest_fb,
                         !edpy->y_0_top);
        egl_texture_blend(edpy->gls, &edpy->blit_fb, &edpy->cursor_fb,
                          !edpy->y_0_top, edpy->pos_x, edpy->pos_y,
                          1.0, 1.0);
    } else {
        /* no cursor -> use simple framebuffer blit */
        egl_fb_blit(&edpy->blit_fb, &edpy->guest_fb, edpy->y_0_top);
    }

    egl_fb_read(edpy->ds, &edpy->blit_fb);
    dpy_gfx_update(edpy->dcl.con, x, y, w, h);
}

static const DisplayChangeListenerOps egl_ops = {
    .dpy_name                = "egl-headless",
    .dpy_refresh             = egl_refresh,
    .dpy_gfx_update          = egl_gfx_update,
    .dpy_gfx_switch          = egl_gfx_switch,

    .dpy_gl_scanout_disable  = egl_scanout_disable,
    .dpy_gl_scanout_texture  = egl_scanout_texture,
#ifdef CONFIG_GBM
    .dpy_gl_scanout_dmabuf   = egl_scanout_dmabuf,
    .dpy_gl_cursor_dmabuf    = egl_cursor_dmabuf,
    .dpy_gl_release_dmabuf   = egl_release_dmabuf,
#endif
    .dpy_gl_cursor_position  = egl_cursor_position,
    .dpy_gl_update           = egl_scanout_flush,
};

static bool
egl_is_compatible_dcl(DisplayGLCtx *dgc,
                      DisplayChangeListener *dcl)
{
    if (!dcl->ops->dpy_gl_update) {
        /*
         * egl-headless is compatible with all 2d listeners, as it blits the GL
         * updates on the 2d console surface.
         */
        return true;
    }

    return dcl->ops == &egl_ops;
}

static const DisplayGLCtxOps eglctx_ops = {
    .dpy_gl_ctx_is_compatible_dcl = egl_is_compatible_dcl,
    .dpy_gl_ctx_create       = egl_create_context,
    .dpy_gl_ctx_destroy      = egl_destroy_context,
    .dpy_gl_ctx_make_current = qemu_egl_make_context_current,
};

static void early_egl_headless_init(DisplayOptions *opts)
{
    DisplayGLMode mode = DISPLAY_GL_MODE_ON;

    if (opts->has_gl) {
        mode = opts->gl;
    }

    egl_init(opts->u.egl_headless.rendernode, mode, &error_fatal);
}

static void egl_headless_init(DisplayState *ds, DisplayOptions *opts)
{
    QemuConsole *con;
    egl_dpy *edpy;
    int idx;

    for (idx = 0;; idx++) {
        DisplayGLCtx *ctx;

        con = qemu_console_lookup_by_index(idx);
        if (!con || !qemu_console_is_graphic(con)) {
            break;
        }

        edpy = g_new0(egl_dpy, 1);
        edpy->dcl.con = con;
        edpy->dcl.ops = &egl_ops;
        edpy->gls = qemu_gl_init_shader();
#ifdef __APPLE__
        for (int i = 0; i < GPU_BUFFERS; i++) {
            edpy->gpu[i].pbuffer = EGL_NO_SURFACE;
        }
#endif
        ctx = g_new0(DisplayGLCtx, 1);
        ctx->ops = &eglctx_ops;
        qemu_console_set_display_gl_ctx(con, ctx);
        register_displaychangelistener(&edpy->dcl);
    }
}

static QemuDisplay qemu_display_egl = {
    .type       = DISPLAY_TYPE_EGL_HEADLESS,
    .early_init = early_egl_headless_init,
    .init       = egl_headless_init,
};

static void register_egl(void)
{
    qemu_display_register(&qemu_display_egl);
}

type_init(register_egl);

module_dep("ui-opengl");
