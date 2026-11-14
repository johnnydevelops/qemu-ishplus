/*
 * QEMU Core Audio playback for iOS (iSH+)
 *
 * coreaudio.m drives macOS's audio devices (the HAL), which iOS does not
 * have: here the guest's sound plays through the RemoteIO audio unit, under
 * the same driver name and options ("coreaudio"). The app sets up the audio
 * session (playback, mixing with other apps). Same ring buffer and jitter
 * buffer as coreaudio.m: the unit's render callback takes what the guest's
 * sound card queued, once half of the ring is there.
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 */

#include <TargetConditionals.h>
#if TARGET_OS_IPHONE

#include "qemu/osdep.h"
#include <AudioToolbox/AudioToolbox.h>
#include <pthread.h>
#include "qemu/module.h"
#include "audio.h"

#define AUDIO_CAP "coreaudio"
#include "audio_int.h"

typedef struct iosVoiceOut {
    HWVoiceOut hw;
    pthread_mutex_t buf_mutex;
    AudioComponentInstance unit;
    uint32_t frames;        /* frames per render callback, as planned */
    bool enabled;
    bool primed;
    bool running;
} iosVoiceOut;

#define IOS_WRAPPER_FUNC(name, ret_type, args_decl, args)      \
    static ret_type glue(ios_, name)args_decl                   \
    {                                                           \
        iosVoiceOut *v = (iosVoiceOut *) hw;                    \
        ret_type ret;                                           \
        pthread_mutex_lock(&v->buf_mutex);                      \
        ret = glue(audio_generic_, name)args;                   \
        pthread_mutex_unlock(&v->buf_mutex);                    \
        return ret;                                             \
    }
IOS_WRAPPER_FUNC(buffer_get_free, size_t, (HWVoiceOut *hw), (hw))
IOS_WRAPPER_FUNC(get_buffer_out, void *, (HWVoiceOut *hw, size_t *size),
                 (hw, size))
IOS_WRAPPER_FUNC(put_buffer_out, size_t,
                 (HWVoiceOut *hw, void *buf, size_t size), (hw, buf, size))
IOS_WRAPPER_FUNC(write, size_t, (HWVoiceOut *hw, void *buf, size_t size),
                 (hw, buf, size))
#undef IOS_WRAPPER_FUNC

/*
 * The unit asks for frames (its own thread, without the BQL): what is
 * queued, or silence while the ring fills up (jitter buffer) or ran dry.
 */
static OSStatus ios_render(void *opaque, AudioUnitRenderActionFlags *flags,
                           const AudioTimeStamp *timestamp, UInt32 bus,
                           UInt32 frames, AudioBufferList *data)
{
    iosVoiceOut *v = opaque;
    HWVoiceOut *hw = &v->hw;
    uint8_t *out = data->mBuffers[0].mData;
    size_t want = (size_t) frames * hw->info.bytes_per_frame;
    size_t pending_frames;

    pthread_mutex_lock(&v->buf_mutex);
    pending_frames = hw->pending_emul / hw->info.bytes_per_frame;
    if (!v->primed && pending_frames < hw->samples / 2) {
        goto silence;
    }
    v->primed = true;
    if (pending_frames < frames) {
        v->primed = false;
        goto silence;
    }
    while (want) {
        size_t start = audio_ring_posb(hw->pos_emul, hw->pending_emul, hw->size_emul);
        size_t len = MIN(MIN(hw->pending_emul, want), hw->size_emul - start);
        memcpy(out, hw->buf_emul + start, len);
        hw->pending_emul -= len;
        want -= len;
        out += len;
    }
    pthread_mutex_unlock(&v->buf_mutex);
    return noErr;

silence:
    pthread_mutex_unlock(&v->buf_mutex);
    memset(data->mBuffers[0].mData, 0, data->mBuffers[0].mDataByteSize);
    *flags |= kAudioUnitRenderAction_OutputIsSilence;
    return noErr;
}

static void ios_update_state(iosVoiceOut *v)
{
    if (!v->unit) {
        return;
    }
    if (v->enabled && !v->running) {
        if (AudioOutputUnitStart(v->unit) == noErr) {
            v->running = true;
        }
    } else if (!v->enabled && v->running) {
        AudioOutputUnitStop(v->unit);
        v->running = false;
    }
}

static int ios_init_out(HWVoiceOut *hw, struct audsettings *as, void *drv_opaque)
{
    iosVoiceOut *v = (iosVoiceOut *) hw;
    Audiodev *dev = drv_opaque;
    AudiodevCoreaudioPerDirectionOptions *cpdo = dev->u.coreaudio.out;
    struct audsettings obt_as = *as;
    AudioComponentDescription desc = {
        .componentType = kAudioUnitType_Output,
        .componentSubType = kAudioUnitSubType_RemoteIO,
        .componentManufacturer = kAudioUnitManufacturer_Apple,
    };
    AudioComponent component;
    OSStatus status;
    uint32_t buffers;

    pthread_mutex_init(&v->buf_mutex, NULL);
    obt_as.fmt = AUDIO_FORMAT_F32;
    audio_pcm_init_info(&hw->info, &obt_as);

    component = AudioComponentFindNext(NULL, &desc);
    if (!component || AudioComponentInstanceNew(component, &v->unit) != noErr) {
        dolog("could not create the RemoteIO audio unit\n");
        v->unit = NULL;
        return -1;
    }

    AudioStreamBasicDescription format = {
        .mSampleRate = hw->info.freq,
        .mFormatID = kAudioFormatLinearPCM,
        .mFormatFlags = kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsPacked,
        .mBytesPerPacket = hw->info.bytes_per_frame,
        .mFramesPerPacket = 1,
        .mBytesPerFrame = hw->info.bytes_per_frame,
        .mChannelsPerFrame = hw->info.nchannels,
        .mBitsPerChannel = hw->info.bits,
    };
    status = AudioUnitSetProperty(v->unit, kAudioUnitProperty_StreamFormat,
                                  kAudioUnitScope_Input, 0, &format, sizeof(format));
    if (status != noErr) {
        dolog("could not set the audio format (%d)\n", (int) status);
        goto fail;
    }
    AURenderCallbackStruct callback = { .inputProc = ios_render, .inputProcRefCon = v };
    status = AudioUnitSetProperty(v->unit, kAudioUnitProperty_SetRenderCallback,
                                  kAudioUnitScope_Input, 0, &callback, sizeof(callback));
    if (status != noErr || AudioUnitInitialize(v->unit) != noErr) {
        dolog("could not set up the audio unit (%d)\n", (int) status);
        goto fail;
    }

    /* The ring: buffer_count callbacks' worth (iOS asks for about 1024
     * frames at a time, at 48 kHz) */
    v->frames = audio_buffer_frames(
        qapi_AudiodevCoreaudioPerDirectionOptions_base(cpdo), &obt_as, 21333);
    buffers = cpdo->has_buffer_count ? cpdo->buffer_count : 4;
    hw->samples = buffers * v->frames;
    audio_generic_initialize_buffer_out(hw);
    return 0;

fail:
    AudioComponentInstanceDispose(v->unit);
    v->unit = NULL;
    return -1;
}

static void ios_fini_out(HWVoiceOut *hw)
{
    iosVoiceOut *v = (iosVoiceOut *) hw;

    if (v->unit) {
        AudioOutputUnitStop(v->unit);
        AudioUnitUninitialize(v->unit);
        AudioComponentInstanceDispose(v->unit);
        v->unit = NULL;
    }
    pthread_mutex_destroy(&v->buf_mutex);
}

static void ios_enable_out(HWVoiceOut *hw, bool enable)
{
    iosVoiceOut *v = (iosVoiceOut *) hw;

    pthread_mutex_lock(&v->buf_mutex);
    v->enabled = enable;
    if (enable) {
        v->primed = false;
    }
    pthread_mutex_unlock(&v->buf_mutex);
    ios_update_state(v);
}

static void *ios_audio_init(Audiodev *dev, Error **errp)
{
    return dev;
}

static void ios_audio_fini(void *opaque)
{
}

static struct audio_pcm_ops ios_pcm_ops = {
    .init_out = ios_init_out,
    .fini_out = ios_fini_out,
    .write = ios_write,
    .buffer_get_free = ios_buffer_get_free,
    .get_buffer_out = ios_get_buffer_out,
    .put_buffer_out = ios_put_buffer_out,
    .enable_out = ios_enable_out,
};

static struct audio_driver ios_audio_driver = {
    .name = "coreaudio",
    .descr = "Core Audio (iOS: RemoteIO)",
    .init = ios_audio_init,
    .fini = ios_audio_fini,
    .pcm_ops = &ios_pcm_ops,
    .max_voices_out = 1,
    .max_voices_in = 0,
    .voice_size_out = sizeof(iosVoiceOut),
    .voice_size_in = 0,
};

/* For iSH+: the guest can have a sound card (this library plays it) */
int ishplus_audio_available(void);
int ishplus_audio_available(void)
{
    return 1;
}

static void register_audio_ios(void)
{
    audio_driver_register(&ios_audio_driver);
}
type_init(register_audio_ios);

#endif /* TARGET_OS_IPHONE */
