// yolo_postprocess.cpp — YOLO output parsing + NMS + drawing for BBV profiling
// No ORT or FFmpeg dependency. Requires stb_image.h for JPEG loading.
// RISC-V binary, runs under QEMU.
//
// Build:
//   g++ -std=c++17 -O2 -I. yolo_postprocess.cpp -o yolo_postprocess
//
// Run:
//   ./yolo_postprocess <output.bin> <image.jpg> [conf] [iou]
//   ./yolo_postprocess --synthetic <image.jpg>    (hardcoded detections for drawing test)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static constexpr int MODEL_INPUT_SIZE = 640;
static constexpr int NUM_ANCHORS  = 8400;
static constexpr int NUM_CLASSES  = 80;
static constexpr int NUM_CHANNELS = 84;

// COCO class names
static const char* COCO_CLASSES[] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

static const uint8_t PALETTE[][3] = {
    {255,0,0}, {0,255,0}, {0,0,255}, {255,255,0},
    {0,255,255}, {255,0,255}, {255,128,0}, {128,0,255},
};

// ---------------------------------------------------------------------------
// Detection + NMS
// ---------------------------------------------------------------------------
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int classId;
};

static float computeIoU(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6f);
}

static std::vector<Detection> nms(std::vector<Detection>& dets, float iouThreshold) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> kept;
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            if (dets[i].classId == dets[j].classId &&
                computeIoU(dets[i], dets[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}

// ---------------------------------------------------------------------------
// YOLO output parsing
// ---------------------------------------------------------------------------
static std::vector<Detection> parseYoloOutput(const float* data,
                                               float confThreshold,
                                               int origW, int origH) {
    float scaleX = static_cast<float>(origW) / MODEL_INPUT_SIZE;
    float scaleY = static_cast<float>(origH) / MODEL_INPUT_SIZE;

    std::vector<Detection> dets;
    for (int a = 0; a < NUM_ANCHORS; a++) {
        float cx = data[0 * NUM_ANCHORS + a];
        float cy = data[1 * NUM_ANCHORS + a];
        float w  = data[2 * NUM_ANCHORS + a];
        float h  = data[3 * NUM_ANCHORS + a];

        float bestScore = 0;
        int bestClass = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = data[(4 + c) * NUM_ANCHORS + a];
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        if (bestScore < confThreshold) continue;

        Detection det;
        det.x1 = (cx - w / 2) * scaleX;
        det.y1 = (cy - h / 2) * scaleY;
        det.x2 = (cx + w / 2) * scaleX;
        det.y2 = (cy + h / 2) * scaleY;
        det.confidence = bestScore;
        det.classId = bestClass;
        dets.push_back(det);
    }
    return dets;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------
static void drawRect(uint8_t* rgb, int stride, int imgW, int imgH,
                      int x1, int y1, int x2, int y2,
                      const uint8_t color[3], int thickness) {
    // Note: x2/y2 are inclusive bounds; caller must ensure they fit within imgW/imgH.
    for (int t = 0; t < thickness; t++) {
        for (int x = x1 + t; x <= x2 - t; x++) {
            if (x >= 0 && x < imgW) {
                if (y1 + t >= 0 && y1 + t < imgH) {
                    int off = ((y1 + t) * stride) + x * 3;
                    rgb[off] = color[0]; rgb[off+1] = color[1]; rgb[off+2] = color[2];
                }
                if (y2 - t >= 0 && y2 - t < imgH) {
                    int off = ((y2 - t) * stride) + x * 3;
                    rgb[off] = color[0]; rgb[off+1] = color[1]; rgb[off+2] = color[2];
                }
            }
        }
        for (int y = y1 + t; y <= y2 - t; y++) {
            if (y >= 0 && y < imgH) {
                if (x1 + t >= 0 && x1 + t < imgW) {
                    int off = (y * stride) + (x1 + t) * 3;
                    rgb[off] = color[0]; rgb[off+1] = color[1]; rgb[off+2] = color[2];
                }
                if (x2 - t >= 0 && x2 - t < imgW) {
                    int off = (y * stride) + (x2 - t) * 3;
                    rgb[off] = color[0]; rgb[off+1] = color[1]; rgb[off+2] = color[2];
                }
            }
        }
    }
}

static void drawDetections(uint8_t* rgb, int stride, int imgW, int imgH,
                            const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        const uint8_t* color = PALETTE[d.classId % 8];
        int x1 = std::max(0, static_cast<int>(d.x1));
        int y1 = std::max(0, static_cast<int>(d.y1));
        int x2 = std::min(imgW - 1, static_cast<int>(d.x2));
        int y2 = std::min(imgH - 1, static_cast<int>(d.y2));
        drawRect(rgb, stride, imgW, imgH, x1, y1, x2, y2, color, 2);

        // Label background
        char label[64];
        snprintf(label, sizeof(label), "%s %.0f%%", COCO_CLASSES[d.classId], d.confidence * 100);
        int labelW = 6;
        for (const char* p = label; *p; p++) labelW += 6;
        int labelH = 12;
        uint8_t bgColor[3] = {
            static_cast<uint8_t>(color[0] * 0.6f),
            static_cast<uint8_t>(color[1] * 0.6f),
            static_cast<uint8_t>(color[2] * 0.6f)
        };
        for (int ly = y1 - labelH; ly < y1; ly++) {
            for (int lx = x1; lx < std::min(x1 + labelW, imgW); lx++) {
                if (ly >= 0 && ly < imgH) {
                    int off = (ly * stride) + lx * 3;
                    rgb[off] = bgColor[0]; rgb[off+1] = bgColor[1]; rgb[off+2] = bgColor[2];
                }
            }
        }

        // Simple text rendering (6x8 font for digits and common chars)
        // For simplicity, just fill the label area with white for now
        // (Full text rendering would need a bitmap font)
    }
}

// ---------------------------------------------------------------------------
// Save PPM
// ---------------------------------------------------------------------------
static bool savePPM(const char* path, const uint8_t* rgb, int w, int h) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, w * h * 3, f);
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// Synthetic detections for drawing verification
// ---------------------------------------------------------------------------
static std::vector<Detection> makeSyntheticDetections(int imgW, int imgH) {
    // Hardcoded realistic YOLO detections scaled to image coordinates.
    // Coordinates are fractions of image dimensions.
    struct RawDet { float rx1, ry1, rx2, ry2; float conf; int cls; };
    RawDet raw[] = {
        {0.10f, 0.15f, 0.55f, 0.90f, 0.923f, 0},  // person
        {0.50f, 0.05f, 0.95f, 0.75f, 0.871f, 5},   // bus
        {0.60f, 0.55f, 0.85f, 0.80f, 0.784f, 1},   // bicycle
        {0.02f, 0.30f, 0.25f, 0.70f, 0.652f, 2},   // car
    };
    std::vector<Detection> dets;
    for (const auto& r : raw) {
        Detection d;
        d.x1 = r.rx1 * imgW;
        d.y1 = r.ry1 * imgH;
        d.x2 = r.rx2 * imgW;
        d.y2 = r.ry2 * imgH;
        d.confidence = r.conf;
        d.classId = r.cls;
        dets.push_back(d);
    }
    return dets;
}

// ---------------------------------------------------------------------------
// Common run logic
// ---------------------------------------------------------------------------
static int runPipeline(const std::vector<Detection>& kept,
                        uint8_t* imgRgb, int imgW, int imgH,
                        double parseTime, double nmsTime) {
    for (size_t i = 0; i < kept.size(); i++) {
        const auto& d = kept[i];
        fprintf(stderr, "  [%zu] %s %.1f%%  box=(%.1f,%.1f,%.1f,%.1f)\n",
                i, COCO_CLASSES[d.classId], d.confidence * 100,
                d.x1, d.y1, d.x2, d.y2);
    }

    auto t4 = std::chrono::steady_clock::now();
    drawDetections(imgRgb, imgW * 3, imgW, imgH, kept);
    auto t5 = std::chrono::steady_clock::now();
    double drawTime = std::chrono::duration<double>(t5 - t4).count();

    const char* resultPath = "result.ppm";
    if (savePPM(resultPath, imgRgb, imgW, imgH)) {
        fprintf(stderr, "Saved: %s (%.3fs to draw)\n", resultPath, drawTime);
    } else {
        fprintf(stderr, "[ERROR] Failed to save %s\n", resultPath);
    }

    double total = parseTime + nmsTime + drawTime;
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Parse: %.3fs (%.0f%%)\n", parseTime, parseTime / total * 100);
    fprintf(stderr, "NMS:   %.3fs (%.0f%%)\n", nmsTime,   nmsTime / total * 100);
    fprintf(stderr, "Draw:  %.3fs (%.0f%%)\n", drawTime,  drawTime / total * 100);
    fprintf(stderr, "Total: %.3fs\n", total);

    return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage:\n"
            "  %s <output.bin> <image.jpg> [conf_thresh] [iou_thresh]\n"
            "  %s --synthetic <image.jpg>\n", argv[0], argv[0]);
        return 1;
    }

    bool synthetic = (strcmp(argv[1], "--synthetic") == 0);

    if (synthetic) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s --synthetic <image.jpg>\n", argv[0]);
            return 1;
        }
        const char* imgPath = argv[2];

        fprintf(stderr, "=== YOLO Postprocess Test (RISC-V) — Synthetic Mode ===\n");
        fprintf(stderr, "Image: %s\n", imgPath);

        int imgW, imgH, imgC;
        uint8_t* imgRgb = stbi_load(imgPath, &imgW, &imgH, &imgC, 3);
        if (!imgRgb) {
            fprintf(stderr, "[ERROR] Cannot load image %s\n", imgPath);
            return 1;
        }
        fprintf(stderr, "Image: %dx%d\n", imgW, imgH);

        auto t0 = std::chrono::steady_clock::now();
        auto dets = makeSyntheticDetections(imgW, imgH);
        auto t1 = std::chrono::steady_clock::now();
        double parseTime = std::chrono::duration<double>(t1 - t0).count();

        fprintf(stderr, "Synthetic detections: %zu (%.3fs)\n", dets.size(), parseTime);

        auto t2 = std::chrono::steady_clock::now();
        auto kept = nms(dets, 0.45f);
        auto t3 = std::chrono::steady_clock::now();
        double nmsTime = std::chrono::duration<double>(t3 - t2).count();

        fprintf(stderr, "After NMS (iou >= 0.45): %zu (%.3fs)\n", kept.size(), nmsTime);

        int rc = runPipeline(kept, imgRgb, imgW, imgH, parseTime, nmsTime);
        stbi_image_free(imgRgb);
        return rc;
    }

    // Normal mode: parse real .bin output
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <output.bin> <image.jpg> [conf_thresh] [iou_thresh]\n", argv[0]);
        return 1;
    }

    const char* binPath = argv[1];
    const char* imgPath = argv[2];
    float confThreshold = argc >= 4 ? atof(argv[3]) : 0.25f;
    float iouThreshold  = argc >= 5 ? atof(argv[4]) : 0.45f;

    fprintf(stderr, "=== YOLO Postprocess Test (RISC-V) ===\n");
    fprintf(stderr, "Output bin: %s\n", binPath);
    fprintf(stderr, "Image:      %s\n", imgPath);
    fprintf(stderr, "Conf:       %.2f\n", confThreshold);
    fprintf(stderr, "IoU:        %.2f\n", iouThreshold);

    FILE* f = fopen(binPath, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", binPath);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    fseek(f, 0, SEEK_SET);

    size_t numFloats = fileSize / sizeof(float);
    fprintf(stderr, "Loaded: %ld bytes = %zu floats\n", fileSize, numFloats);

    if (numFloats != (size_t)(NUM_CHANNELS * NUM_ANCHORS)) {
        fprintf(stderr, "[WARN] Expected %d floats (84x8400), got %zu\n",
                NUM_CHANNELS * NUM_ANCHORS, numFloats);
    }

    std::vector<float> outputData(numFloats);
    if (fread(outputData.data(), sizeof(float), numFloats, f) != numFloats) {
        fprintf(stderr, "[ERROR] Failed to read data\n");
        fclose(f);
        return 1;
    }
    fclose(f);

    int imgW, imgH, imgC;
    uint8_t* imgRgb = stbi_load(imgPath, &imgW, &imgH, &imgC, 3);
    if (!imgRgb) {
        fprintf(stderr, "[ERROR] Cannot load image %s\n", imgPath);
        return 1;
    }
    fprintf(stderr, "Image: %dx%d\n", imgW, imgH);

    auto t0 = std::chrono::steady_clock::now();
    auto dets = parseYoloOutput(outputData.data(), confThreshold, imgW, imgH);
    auto t1 = std::chrono::steady_clock::now();
    double parseTime = std::chrono::duration<double>(t1 - t0).count();

    fprintf(stderr, "Raw detections (conf >= %.2f): %zu (%.3fs)\n",
            confThreshold, dets.size(), parseTime);

    auto t2 = std::chrono::steady_clock::now();
    auto kept = nms(dets, iouThreshold);
    auto t3 = std::chrono::steady_clock::now();
    double nmsTime = std::chrono::duration<double>(t3 - t2).count();

    fprintf(stderr, "After NMS (iou >= %.2f): %zu (%.3fs)\n",
            iouThreshold, kept.size(), nmsTime);

    int rc = runPipeline(kept, imgRgb, imgW, imgH, parseTime, nmsTime);
    stbi_image_free(imgRgb);
    return rc;
}
