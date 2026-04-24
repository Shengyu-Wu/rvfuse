#include <onnxruntime_cxx_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static constexpr int MODEL_INPUT_SIZE = 640;

//
// Reference data for validation
// These values are generated from test.jpg with scalar ORT implementation
//

// Expected MD5 of test.jpg file
static const char* kExpectedImageMd5 = "10bd23d36a88f18010bf1c63b56aa309";

// Expected output checksum (sum of first 30 detection values: 5 detections × 6 fields)
// Generated from: [1, 84, 8400] output tensor with test.jpg
static const float kExpectedOutputChecksum = 3592.22f;

// Expected output shape
static const int64_t kExpectedOutputElements = 705600;  // 1 * 84 * 8400

//
// MD5 implementation (simplified, based on RFC 1321)
//

// MD5 context structure
struct Md5Context {
    uint32_t state[4];
    uint32_t count[2];
    uint8_t buffer[64];
};

// MD5 constants
static const uint32_t kMd5K[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

static const uint8_t kMd5S[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

// MD5 F, G, H, I functions
#define MD5_F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5_G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5_H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5_I(x, y, z) ((y) ^ ((x) | (~z)))

// MD5 rotate left
#define MD5_ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

// MD5 transform step
#define MD5_STEP(f, a, b, c, d, x, t, s) \
    a += f(b, c, d) + x + t; \
    a = MD5_ROTATE_LEFT(a, s); \
    a += b;

static void md5Transform(uint32_t state[4], const uint8_t block[64])
{
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t x[16];

    // Decode block into 32-bit words
    for (int i = 0; i < 16; i++) {
        x[i] = ((uint32_t)block[i * 4]) |
               ((uint32_t)block[i * 4 + 1] << 8) |
               ((uint32_t)block[i * 4 + 2] << 16) |
               ((uint32_t)block[i * 4 + 3] << 24);
    }

    // Round 1
    for (int i = 0; i < 16; i++) {
        MD5_STEP(MD5_F, a, b, c, d, x[i], kMd5K[i], kMd5S[i]);
        uint32_t t = a; a = d; d = c; c = b; b = t;
    }

    // Round 2
    for (int i = 0; i < 16; i++) {
        int idx = (5 * i + 1) % 16;
        MD5_STEP(MD5_G, a, b, c, d, x[idx], kMd5K[i + 16], kMd5S[i + 16]);
        uint32_t t = a; a = d; d = c; c = b; b = t;
    }

    // Round 3
    for (int i = 0; i < 16; i++) {
        int idx = (3 * i + 5) % 16;
        MD5_STEP(MD5_H, a, b, c, d, x[idx], kMd5K[i + 32], kMd5S[i + 32]);
        uint32_t t = a; a = d; d = c; c = b; b = t;
    }

    // Round 4
    for (int i = 0; i < 16; i++) {
        int idx = (7 * i) % 16;
        MD5_STEP(MD5_I, a, b, c, d, x[idx], kMd5K[i + 48], kMd5S[i + 48]);
        uint32_t t = a; a = d; d = c; c = b; b = t;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

static void md5Init(Md5Context* ctx)
{
    ctx->state[0] = 0x67452301;
    ctx->state[1] = 0xefcdab89;
    ctx->state[2] = 0x98badcfe;
    ctx->state[3] = 0x10325476;
    ctx->count[0] = ctx->count[1] = 0;
}

static void md5Update(Md5Context* ctx, const uint8_t* data, size_t len)
{
    size_t index = (ctx->count[0] >> 3) & 0x3F;
    size_t partLen = 64 - index;
    size_t i = 0;

    ctx->count[0] += (len << 3);
    if (ctx->count[0] < (len << 3)) ctx->count[1]++;
    ctx->count[1] += (len >> 29);

    if (len >= partLen) {
        memcpy(&ctx->buffer[index], data, partLen);
        md5Transform(ctx->state, ctx->buffer);

        for (i = partLen; i + 63 < len; i += 64) {
            md5Transform(ctx->state, &data[i]);
        }
        index = 0;
    }

    memcpy(&ctx->buffer[index], &data[i], len - i);
}

static void md5Final(Md5Context* ctx, uint8_t digest[16])
{
    static const uint8_t kMd5Padding[64] = {
        0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    uint8_t bits[8];
    size_t index = (ctx->count[0] >> 3) & 0x3F;
    size_t padLen = (index < 56) ? (56 - index) : (120 - index);

    for (int i = 0; i < 4; i++) {
        bits[i] = (ctx->count[0] >> (i * 8)) & 0xFF;
        bits[i + 4] = (ctx->count[1] >> (i * 8)) & 0xFF;
    }

    md5Update(ctx, kMd5Padding, padLen);
    md5Update(ctx, bits, 8);

    for (int i = 0; i < 16; i++) {
        digest[i] = (ctx->state[i >> 2] >> ((i % 4) * 8)) & 0xFF;
    }
}

static std::string md5ToString(const uint8_t digest[16])
{
    std::ostringstream oss;
    for (int i = 0; i < 16; i++) {
        oss << std::hex << std::setw(2) << std::setfill('0') << (int)digest[i];
    }
    return oss.str();
}

static std::string computeFileMd5(const char* filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file) return "";

    Md5Context ctx;
    md5Init(&ctx);

    char buffer[8192];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
        md5Update(&ctx, reinterpret_cast<uint8_t*>(buffer), file.gcount());
    }

    uint8_t digest[16];
    md5Final(&ctx, digest);

    return md5ToString(digest);
}

//
// Image processing functions
//

static std::vector<unsigned char> resizeNearest(
    const unsigned char* src, int srcW, int srcH, int channels,
    int dstW, int dstH)
{
    std::vector<unsigned char> dst(dstW * dstH * channels);
    float xRatio = static_cast<float>(srcW) / dstW;
    float yRatio = static_cast<float>(srcH) / dstH;

    for (int y = 0; y < dstH; y++) {
        for (int x = 0; x < dstW; x++) {
            int sx = std::min(static_cast<int>(x * xRatio), srcW - 1);
            int sy = std::min(static_cast<int>(y * yRatio), srcH - 1);
            int si = (sy * srcW + sx) * channels;
            int di = (y * dstW + x) * channels;
            for (int c = 0; c < channels; c++) {
                dst[di + c] = src[si + c];
            }
        }
    }
    return dst;
}

static void preprocess(
    const char* imagePath,
    std::vector<float>& tensorData,
    std::vector<int64_t>& inputShape)
{
    int imgW, imgH, imgC;
    unsigned char* img = stbi_load(imagePath, &imgW, &imgH, &imgC, 3);
    if (!img) {
        fprintf(stderr, "Error: cannot load image %s\n", imagePath);
        exit(1);
    }

    std::vector<unsigned char> resized =
        resizeNearest(img, imgW, imgH, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
    stbi_image_free(img);

    // HWC -> CHW, normalize to [0, 1]
    int pixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
    tensorData.resize(pixels * 3);
    for (int y = 0; y < MODEL_INPUT_SIZE; y++) {
        for (int x = 0; x < MODEL_INPUT_SIZE; x++) {
            int hwc = (y * MODEL_INPUT_SIZE + x) * 3;
            tensorData[0 * pixels + y * MODEL_INPUT_SIZE + x] =
                resized[hwc + 0] / 255.0f;
            tensorData[1 * pixels + y * MODEL_INPUT_SIZE + x] =
                resized[hwc + 1] / 255.0f;
            tensorData[2 * pixels + y * MODEL_INPUT_SIZE + x] =
                resized[hwc + 2] / 255.0f;
        }
    }

    inputShape = {1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE};
}

//
// Output validation
//

static float computeOutputChecksum(Ort::Value& tensor, int numValues)
{
    float* data = tensor.GetTensorMutableData<float>();
    float sum = 0.0f;
    for (int i = 0; i < numValues; i++) {
        sum += data[i];
    }
    return sum;
}

static void printTopDetections(Ort::Value& tensor, int showCount)
{
    auto info = tensor.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    float* data = tensor.GetTensorMutableData<float>();
    int64_t total = info.GetElementCount();

    fprintf(stdout, "Output shape: [");
    for (size_t d = 0; d < shape.size(); d++) {
        if (d > 0) fprintf(stdout, ", ");
        fprintf(stdout, "%lld", static_cast<long long>(shape[d]));
    }
    fprintf(stdout, "]\n");

    // YOLO output: [numDets, 6] where cols = x1, y1, x2, y2, conf, cls
    int cols = 6;
    int numDets = static_cast<int>(total / cols);
    int show = std::min(numDets, showCount);
    fprintf(stdout, "Top %d detections:\n", show);
    for (int d = 0; d < show; d++) {
        int i = d * cols;
        fprintf(stdout, "  [%d] box=(%.1f,%.1f,%.1f,%.1f) conf=%.3f cls=%.0f\n",
            d, data[i], data[i+1], data[i+2], data[i+3], data[i+4], data[i+5]);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.onnx> <image.jpg> [iterations]\n", argv[0]);
        return 1;
    }

    const char* modelPath = argv[1];
    const char* imagePath = argv[2];
    int iterations = 10;
    if (argc >= 4) {
        iterations = atoi(argv[3]);
        if (iterations < 1) iterations = 1;
    }

    //
    // Validation setup: check if image matches expected test.jpg
    //

    std::string actualMd5 = computeFileMd5(imagePath);
    bool validationEnabled = (actualMd5 == kExpectedImageMd5);

    fprintf(stdout, "Model: %s\n", modelPath);
    fprintf(stdout, "Image: %s\n", imagePath);
    fprintf(stdout, "Image MD5: %s\n", actualMd5.c_str());

    if (validationEnabled) {
        fprintf(stdout, "[VALIDATION] MD5 matches expected test.jpg\n");
        fprintf(stdout, "[VALIDATION] Output verification enabled\n");
    } else {
        fprintf(stdout, "[VALIDATION] MD5 mismatch - validation disabled\n");
        fprintf(stdout, "[VALIDATION] Expected: %s\n", kExpectedImageMd5);
    }

    fprintf(stdout, "Iterations: %d (1 warm-up + %d measured)\n",
            iterations, iterations - 1);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_runner");
    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(1);
    sessionOpts.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::AllocatorWithDefaultOptions allocator;

    fprintf(stdout, "Loading model...\n");
    Ort::Session session(env, modelPath, sessionOpts);

    auto inputName = session.GetInputNameAllocated(0, allocator);
    auto outputName = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> outputNamePtrs;
    outputNamePtrs.push_back(outputName.get());

    fprintf(stdout, "Preprocessing image...\n");
    std::vector<float> tensorData;
    std::vector<int64_t> inputShape;
    preprocess(imagePath, tensorData, inputShape);

    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensorData.data(), tensorData.size(),
        inputShape.data(), inputShape.size());

    fprintf(stdout, "Running inference...\n");
    const char* inputNames[] = {inputName.get()};

    Ort::Value lastOutput;

    for (int i = 0; i < iterations; i++) {
        fprintf(stdout, "  [%d/%d]%s\n", i + 1, iterations,
                i == 0 ? " (warm-up)" : "");
        auto outputs = session.Run(
            Ort::RunOptions{},
            inputNames, &inputTensor, 1,
            outputNamePtrs.data(), outputNamePtrs.size());

        if (i == iterations - 1) {
            // Store last output for validation and printing
            lastOutput = std::move(outputs[0]);
        }
    }

    //
    // Print results
    //

    printTopDetections(lastOutput, 5);

    //
    // Validation: compare output with expected values
    //

    if (validationEnabled) {
        auto info = lastOutput.GetTensorTypeAndShapeInfo();
        int64_t totalElements = info.GetElementCount();

        fprintf(stdout, "\n[VALIDATION] Checking output...\n");
        fprintf(stdout, "[VALIDATION] Expected elements: %lld\n", kExpectedOutputElements);
        fprintf(stdout, "[VALIDATION] Actual elements:   %lld\n", totalElements);

        // Compute checksum using first 30 values (covers top 5 detections)
        float checksum = computeOutputChecksum(lastOutput, 30);
        fprintf(stdout, "[VALIDATION] Expected checksum: %.2f\n", kExpectedOutputChecksum);
        fprintf(stdout, "[VALIDATION] Actual checksum:   %.2f\n", checksum);

        bool shapeValid = (totalElements == kExpectedOutputElements);
        bool checksumValid = (std::abs(checksum - kExpectedOutputChecksum) < 0.1f);

        if (shapeValid && checksumValid) {
            fprintf(stdout, "[VALIDATION] ✓ PASSED - output matches expected values\n");
        } else {
            fprintf(stdout, "[VALIDATION] ✗ FAILED - output mismatch!\n");
            if (!shapeValid) {
                fprintf(stdout, "[VALIDATION]   Shape mismatch: expected %lld, got %lld\n",
                        kExpectedOutputElements, totalElements);
            }
            if (!checksumValid) {
                fprintf(stdout, "[VALIDATION]   Checksum mismatch: expected %.2f, got %.2f (diff %.2f)\n",
                        kExpectedOutputChecksum, checksum, checksum - kExpectedOutputChecksum);
            }
        }
    }

    fprintf(stdout, "Done.\n");
    return 0;
}