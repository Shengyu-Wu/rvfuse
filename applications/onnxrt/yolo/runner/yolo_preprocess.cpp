// yolo_preprocess.cpp — FFmpeg video decode + image preprocessing for BBV profiling
// No ORT dependency. Requires FFmpeg (libavcodec/libavformat/libswscale/libavutil).
// RISC-V binary, runs under QEMU.
//
// Build (inside RISC-V Docker):
//   g++ -std=c++17 -O2 yolo_preprocess.cpp \
//       -o yolo_preprocess \
//       -lavcodec -lavformat -lswscale -lavutil
//
// Run:
//   qemu-riscv64 -L <sysroot> ./yolo_preprocess <input.mp4> [max_frames]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

static constexpr int MODEL_INPUT_SIZE = 640;
static constexpr int INPUT_CHANNELS   = 3;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input.mp4> [max_frames]\n", argv[0]);
        return 1;
    }

    const char* inputPath = argv[1];
    int maxFrames = argc >= 3 ? atoi(argv[2]) : 5;
    if (maxFrames <= 0) maxFrames = 5;

    fprintf(stderr, "=== YOLO Preprocess Test (RISC-V) ===\n");
    fprintf(stderr, "Input:  %s\n", inputPath);
    fprintf(stderr, "Frames: %d\n", maxFrames);

    // Open input
    AVFormatContext* inFmtCtx = nullptr;
    int ret = avformat_open_input(&inFmtCtx, inputPath, nullptr, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[ERROR] Cannot open input\n");
        return 1;
    }

    ret = avformat_find_stream_info(inFmtCtx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "[ERROR] Cannot find stream info\n");
        return 1;
    }

    int streamIdx = av_find_best_stream(inFmtCtx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (streamIdx < 0) {
        fprintf(stderr, "[ERROR] No video stream\n");
        return 1;
    }

    const AVCodec* codec = avcodec_find_decoder(inFmtCtx->streams[streamIdx]->codecpar->codec_id);
    AVCodecContext* codecCtx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codecCtx, inFmtCtx->streams[streamIdx]->codecpar);
    avcodec_open2(codecCtx, codec, nullptr);

    int srcW = codecCtx->width;
    int srcH = codecCtx->height;
    fprintf(stderr, "Stream: %dx%d\n", srcW, srcH);

    // SwsContext for resize (src RGB24 → dst 640x640 RGB24)
    SwsContext* swsResize = sws_getContext(
        srcW, srcH, AV_PIX_FMT_RGB24,
        MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!swsResize) {
        fprintf(stderr, "[ERROR] sws_getContext failed for resize\n");
        return 1;
    }

    // Pre-allocate buffers
    std::vector<uint8_t> rgb640(MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 3);
    std::vector<float> tensorData(INPUT_CHANNELS * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);

    AVPacket* pkt = av_packet_alloc();
    int frameCount = 0;

    double totalDecode = 0, totalRgb = 0, totalResize = 0, totalNorm = 0;

    fprintf(stderr, "\n--- Processing ---\n");

    while (frameCount < maxFrames) {
        // Decode next frame
        AVFrame* frame = nullptr;
        while (true) {
            ret = av_read_frame(inFmtCtx, pkt);
            if (ret < 0) { av_packet_unref(pkt); goto done; }
            if (pkt->stream_index != streamIdx) {
                av_packet_unref(pkt);
                continue;
            }
            avcodec_send_packet(codecCtx, pkt);
            av_packet_unref(pkt);
            frame = av_frame_alloc();
            ret = avcodec_receive_frame(codecCtx, frame);
            if (ret == 0) break;
            av_frame_free(&frame);
            if (ret == AVERROR(EAGAIN)) continue;
            av_packet_unref(pkt);
            goto done;
        }

        // Step 1: Decode → RGB24
        auto t0 = std::chrono::steady_clock::now();

        AVFrame* rgbFrame = av_frame_alloc();
        rgbFrame->format = AV_PIX_FMT_RGB24;
        rgbFrame->width = srcW;
        rgbFrame->height = srcH;
        av_frame_get_buffer(rgbFrame, 0);

        auto t1 = std::chrono::steady_clock::now();

        SwsContext* toRgb = sws_getContext(
            frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
            srcW, srcH, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!toRgb) {
            fprintf(stderr, "[WARN] sws_getContext failed for frame %d, skipping\n", frameCount + 1);
            av_frame_unref(frame);
            continue;
        }
        sws_scale(toRgb, frame->data, frame->linesize, 0, frame->height,
                  rgbFrame->data, rgbFrame->linesize);
        sws_freeContext(toRgb);

        auto t2 = std::chrono::steady_clock::now();

        // Step 2: Resize to 640x640
        uint8_t* dstSlice[1] = { rgb640.data() };
        int dstStride[1] = { MODEL_INPUT_SIZE * 3 };
        sws_scale(swsResize, rgbFrame->data, rgbFrame->linesize, 0, srcH,
                  dstSlice, dstStride);
        av_frame_free(&rgbFrame);

        auto t3 = std::chrono::steady_clock::now();

        // Step 3: HWC → CHW, normalize
        int pixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
        for (int y = 0; y < MODEL_INPUT_SIZE; y++) {
            for (int x = 0; x < MODEL_INPUT_SIZE; x++) {
                int hwc = (y * MODEL_INPUT_SIZE + x) * 3;
                tensorData[0 * pixels + y * MODEL_INPUT_SIZE + x] = rgb640[hwc + 0] / 255.0f;
                tensorData[1 * pixels + y * MODEL_INPUT_SIZE + x] = rgb640[hwc + 1] / 255.0f;
                tensorData[2 * pixels + y * MODEL_INPUT_SIZE + x] = rgb640[hwc + 2] / 255.0f;
            }
        }

        auto t4 = std::chrono::steady_clock::now();

        double dtDecode = std::chrono::duration<double>(t1 - t0).count();
        double dtRgb    = std::chrono::duration<double>(t2 - t1).count();
        double dtResize = std::chrono::duration<double>(t3 - t2).count();
        double dtNorm   = std::chrono::duration<double>(t4 - t3).count();
        totalDecode += dtDecode;
        totalRgb    += dtRgb;
        totalResize += dtResize;
        totalNorm   += dtNorm;

        // Validate tensor
        float minVal = tensorData[0], maxVal = tensorData[0], sumVal = 0;
        for (int i = 0; i < INPUT_CHANNELS * pixels; i++) {
            if (tensorData[i] < minVal) minVal = tensorData[i];
            if (tensorData[i] > maxVal) maxVal = tensorData[i];
            sumVal += tensorData[i];
        }
        float meanVal = sumVal / (INPUT_CHANNELS * pixels);

        frameCount++;
        fprintf(stderr, "Frame %d/%d: %dx%d → %dx%d | "
                "decode %.3fs rgb %.3fs resize %.3fs norm %.3fs | "
                "tensor min=%.3f max=%.3f mean=%.3f\n",
                frameCount, maxFrames, srcW, srcH, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
                dtDecode, dtRgb, dtResize, dtNorm,
                minVal, maxVal, meanVal);

        // Save first frame tensor for comparison
        if (frameCount == 1) {
            const char* savePath = "preprocessed_frame1.bin";
            FILE* f = fopen(savePath, "wb");
            if (f) {
                fwrite(tensorData.data(), sizeof(float), tensorData.size(), f);
                fclose(f);
                fprintf(stderr, "  Saved tensor to %s (%zu floats)\n",
                        savePath, tensorData.size());
            }
        }

        av_frame_free(&frame);
    }

done:
    av_packet_free(&pkt);
    sws_freeContext(swsResize);
    avcodec_free_context(&codecCtx);
    avformat_close_input(&inFmtCtx);

    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Frames: %d\n", frameCount);
    if (frameCount > 0) {
        double total = totalDecode + totalRgb + totalResize + totalNorm;
        fprintf(stderr, "Total:   %.3fs (%.1f fps)\n", total, frameCount / total);
        fprintf(stderr, "Decode:  %.3fs (%.0f%%)\n", totalDecode, totalDecode / total * 100);
        fprintf(stderr, "RGB:     %.3fs (%.0f%%)\n", totalRgb,    totalRgb / total * 100);
        fprintf(stderr, "Resize:  %.3fs (%.0f%%)\n", totalResize, totalResize / total * 100);
        fprintf(stderr, "Norm:    %.3fs (%.0f%%)\n", totalNorm,   totalNorm / total * 100);
    }

    return 0;
}
