/*
 * Entry point for hosts that run QEMU inside their own process, such as iOS
 * apps, which cannot start other programs. With -Dembedded=true each system
 * emulator is built as libqemu-system-<target> exporting this function
 * instead of an executable with main().
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu-main.h"
#include "qemu/main-loop.h"
#include "system/replay.h"
#include "system/system.h"

/* Referenced by UI code that wants the main thread; unused when embedded. */
int (*qemu_main)(void);

/*
 * Runs a VM to completion on the calling thread, which becomes QEMU's main
 * loop thread. Returns the exit status. QEMU keeps global state, so this can
 * be called once per process.
 */
__attribute__((visibility("default")))
int qemu_embedded_main(int argc, char **argv)
{
    int status;

    /* qemu_init returns with the BQL and the replay mutex held */
    qemu_init(argc, argv);
    status = qemu_main_loop();
    qemu_cleanup(status);
    bql_unlock();
    replay_mutex_unlock();
    return status;
}
