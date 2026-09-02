// sd-decoder: standalone VAE decoder for .lat latent files.
//
//   sd-decoder --vae <file> --latent-file x.lat -o out.png
//   sd-decoder --vae <file> -W 1024 -H 1024 -o noise.png [--seed 1]
//
// Loads ONLY the VAE (no diffusion model, no text encoder), decodes a
// saved .lat latent (or a random blob shaped like one) and writes the
// image. The model family used by the producing pipeline is read from the
// .lat v2 header; when absent (v1 file or random mode) pass --model-family
// (sd_version_to_str names, e.g. "Flux.2 klein", "Flux", "SDXL"). Fallback
// when nothing is known: "Flux.2 klein".
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <random>
#include <string>
#include <vector>

#if SD_DECODER_HAS_GLFW
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/gl.h>
#else
#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#else
#include <sys/select.h>
#include <unistd.h>
#endif
#endif

#include "stable-diffusion.h"
#include "common/media_io.h"

// Set by the Ctrl-C handler so the dream loop can wind down cleanly.
static std::atomic<bool> g_quit_interactive{false};

struct Options {
    std::string vae_path;
    std::string latent_path;  // empty = random blob mode
    std::string output_path   = "output.png";
    int width                 = 512;
    int height                = 512;
    int channels_override     = 0;  // 0 = auto from VAE weights
    int model_family          = -1; // SDVersion; -1 = from .lat header, else default
    uint64_t seed             = 0;  // 0 = seed from entropy
    int n_threads             = 0;  // 0 = lib default
    bool gaussian             = true;  // noise marginal: gaussian (N(0,1)) or uniform [-1,1]
    int blur                  = 5;     // box-blur radius over latent cells (dreamy default)
    double amp                = 3.0;   // target per-channel std after blur+renormalize (dreamy default)
    bool loop                 = false; // interactive: keypress = new dream, Ctrl-C = quit
    int iterations            = 0;     // bounded loop mode: N generations then exit (0 = off)
};

static void print_usage(const char* argv0) {
    fprintf(stderr,
            "usage: %s --vae <vae-file> [--latent-file <x.lat>] [-o out.png]\n"
            "       [-W width] [-H height] [--channels N] [--model-family NAME]\n"
            "       [--seed N] [--n-threads N] [-v]\n"
            "\n"
            "  --vae <file>          VAE weights (safetensors/gguf); required\n"
            "  --latent-file <file>  decode a saved .lat latent (SDLT format)\n"
            "  -o, --output <file>   output image path (default output.png)\n"
            "  -W/-H                 canvas for random-blob mode (default 512x512)\n"
            "  --channels N          latent channels for random mode (default: read from VAE)\n"
            "  --model-family NAME   model family when the .lat has no version header,\n"
            "                        e.g. \"Flux.2 klein\", \"Flux.2\", \"Flux\", \"SDXL\"\n"
            "  --seed N              random-blob seed; omitted/0 = fresh OS-entropy seed every run\n"
            "  --noise MODE          noise marginal for random mode: gaussian (N(0,1), default) or uniform\n"
            "  --blur N              box-blur radius over latent cells (default 5). Blurring injects\n"
            "                        the low-frequency spatial structure real latents have -> the decode\n"
            "                        reads as soft image-like fields instead of static\n"
            "  --amp S               target per-channel std after blur + renormalize (default 3.0).\n"
            "                        Higher values push the decoder harder -> saturated, dreamier colors\n"
            "                        (--blur 0 --amp 1.0 reproduces the old pure-static look)\n"
            "  --loop                interactive random mode: opens a GLFW window sized to the\n"
            "                        output (-W/-H, default 512x512). VAE loads once; SPACE = one\n"
            "                        dream, P = continuous autoplay (title shows decode throughput\n"
            "                        in img/s), ESC = quit. No file output in window mode\n"
            "  --iterations N        bounded loop: generate N dream images back-to-back then exit\n",
            argv0);
}

// ---- .lat reader (mirror of the lib's read_latent_file; format frozen) ----
static bool read_lat_file(const std::string& path,
                          std::vector<float>& data,
                          std::vector<int64_t>& shape,
                          int& model_version) {
    model_version = -1;
    FILE* f       = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "sd-decoder: cannot open '%s'\n", path.c_str());
        return false;
    }
    bool ok = false;
    do {
        char magic[4] = {0};
        uint32_t version = 0, dtype = 0, ndim = 0, dims[4] = {0, 0, 0, 0};
        uint64_t nbytes = 0;
        if (fread(magic, 1, 4, f) != 4) break;
        if (memcmp(magic, "SDLT", 4) != 0) {
            fprintf(stderr, "sd-decoder: '%s' is not a .lat file (bad magic)\n", path.c_str());
            break;
        }
        if (fread(&version, sizeof(version), 1, f) != 1) break;
        if (fread(&dtype, sizeof(dtype), 1, f) != 1) break;
        if (fread(&ndim, sizeof(ndim), 1, f) != 1) break;
        if (fread(dims, sizeof(dims), 1, f) != 1) break;
        if (fread(&nbytes, sizeof(nbytes), 1, f) != 1) break;
        if (version == 2) {
            uint32_t mv = 0;
            if (fread(&mv, sizeof(mv), 1, f) != 1) break;
            model_version = static_cast<int>(mv);
        } else if (version != 1) {
            fprintf(stderr, "sd-decoder: unsupported .lat version %u\n", version);
            break;
        }
        if (dtype != 0 || ndim == 0 || ndim > 4) {
            fprintf(stderr, "sd-decoder: unsupported .lat header\n");
            break;
        }
        shape.clear();
        int64_t nelem = 1;
        for (uint32_t i = 0; i < ndim; i++) {
            shape.push_back(static_cast<int64_t>(dims[i]));
            nelem *= static_cast<int64_t>(dims[i]);
        }
        if (nbytes != static_cast<uint64_t>(nelem) * sizeof(float) || nelem <= 0) {
            fprintf(stderr, "sd-decoder: .lat size mismatch\n");
            break;
        }
        data.resize(static_cast<size_t>(nelem));
        if (fread(data.data(), sizeof(float), static_cast<size_t>(nelem), f) != static_cast<size_t>(nelem)) {
            fprintf(stderr, "sd-decoder: short read on '%s'\n", path.c_str());
            break;
        }
        ok = true;
    } while (0);
    fclose(f);
    return ok;
}

static void log_cb(enum sd_log_level_t level, const char* log, void* data) {
    (void)level;
    (void)data;
    fputs(log, stderr);
}

#if SD_DECODER_HAS_GLFW
// ---- GLFW dream window (--loop) ----
// The VAE stays loaded. SPACE decodes one random dream and draws it as a
// fullscreen texture; P toggles continuous autoplay (title shows decode
// throughput in img/s over a rolling 2s window); ESC quits.
static std::atomic<bool> g_glfw_regen{false};
static std::atomic<bool> g_glfw_autoplay{false};  // P: continuous generation

static void glfw_key_callback(GLFWwindow* window, int key, int /*scancode*/, int action, int /*mods*/) {
    if (action != GLFW_PRESS) {
        return;
    }
    switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
        case GLFW_KEY_SPACE:
            g_glfw_regen.store(true);  // one dream per press
            break;
        case GLFW_KEY_P:
            g_glfw_autoplay.store(!g_glfw_autoplay.load());
            if (g_glfw_autoplay.load()) {
                g_glfw_regen.store(true);  // start immediately
                fprintf(stderr, "sd-decoder: autoplay ON (P to stop)\n");
            } else {
                fprintf(stderr, "sd-decoder: autoplay OFF - SPACE for single dreams\n");
            }
            break;
        default:
            break;  // other keys ignored
    }
}

static void glfw_framebuffer_size_callback(GLFWwindow* /*window*/, int width, int height) {
    glViewport(0, 0, width, height);
}

// ---- textured fullscreen quad (LearnOpenGL 4.1 style, adapted to fullscreen) ----

static const char* k_vertex_shader_src = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

static const char* k_fragment_shader_src = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D ourTexture;
void main() {
    FragColor = texture(ourTexture, TexCoord);
}
)";

static GLuint s_quad_vao = 0;
static GLuint s_quad_vbo = 0;
static GLuint s_tex     = 0;
static GLuint s_prog    = 0;

static bool gl_log_shader(GLuint shader, const char* name) {
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok) {
        return true;
    }
    char log[1024];
    glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
    fprintf(stderr, "sd-decoder: %s shader compile error:\n%s\n", name, log);
    return false;
}

static bool init_texture_quad() {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &k_vertex_shader_src, nullptr);
    glCompileShader(vs);
    if (!gl_log_shader(vs, "vertex")) return false;
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &k_fragment_shader_src, nullptr);
    glCompileShader(fs);
    if (!gl_log_shader(fs, "fragment")) return false;

    s_prog = glCreateProgram();
    glAttachShader(s_prog, vs);
    glAttachShader(s_prog, fs);
    glLinkProgram(s_prog);
    GLint ok = 0;
    glGetProgramiv(s_prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(s_prog, sizeof(log), nullptr, log);
        fprintf(stderr, "sd-decoder: program link error:\n%s\n", log);
        return false;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);

    // Fullscreen quad (two triangles covering NDC [-1,1]). Texcoord v is flipped
    // (v=0 at the TOP of the screen) because the decoded image rows are top-down
    // while OpenGL textures start at v=0 on the bottom row.
    const float quad[] = {
        // positions      // texcoords
        -1.0f, -1.0f,     0.0f, 1.0f,
         1.0f, -1.0f,     1.0f, 1.0f,
         1.0f,  1.0f,     1.0f, 0.0f,
         1.0f,  1.0f,     1.0f, 0.0f,
        -1.0f,  1.0f,     0.0f, 0.0f,
        -1.0f, -1.0f,     0.0f, 1.0f,
    };
    glGenVertexArrays(1, &s_quad_vao);
    glGenBuffers(1, &s_quad_vbo);
    glBindVertexArray(s_quad_vao);
    glBindBuffer(GL_ARRAY_BUFFER, s_quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glGenTextures(1, &s_tex);
    glBindTexture(GL_TEXTURE_2D, s_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // 1x1 placeholder until the first generation lands.
    const unsigned char black[3] = {0, 0, 0};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, black);
    glUseProgram(s_prog);
    glUniform1i(glGetUniformLocation(s_prog, "ourTexture"), 0);
    return true;
}

static void upload_image_to_texture(const sd_image_t& image) {
    glBindTexture(GL_TEXTURE_2D, s_tex);
    const GLenum fmt  = image.channel == 4 ? GL_RGBA : GL_RGB;
    const GLint  ifmt = image.channel == 4 ? GL_RGBA8 : GL_RGB8;
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt,
                 static_cast<GLsizei>(image.width), static_cast<GLsizei>(image.height),
                 0, fmt, GL_UNSIGNED_BYTE, image.data);
}

static int glfw_dream_loop(int window_width,
                           int window_height,
                           uint64_t fixed_seed,
                           const std::function<void(uint64_t)>& build_random,
                           const std::function<sd_image_t()>& decode_only) {
    if (!glfwInit()) {
        fprintf(stderr, "sd-decoder: glfwInit failed (no display session?)\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    char title[128];
    snprintf(title, sizeof(title), "sd-decoder dream %dx%d", window_width, window_height);
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, title, nullptr, nullptr);
    if (window == nullptr) {
        fprintf(stderr, "sd-decoder: failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, glfw_framebuffer_size_callback);
    glfwSetKeyCallback(window, glfw_key_callback);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        fprintf(stderr, "sd-decoder: failed to initialize GLAD\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    glViewport(0, 0, window_width, window_height);
    glfwSwapInterval(1);

    if (!init_texture_quad()) {
        fprintf(stderr, "sd-decoder: GL setup failed\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    g_glfw_regen.store(false);
    g_glfw_autoplay.store(false);
    fprintf(stderr,
            "sd-decoder: dream window open (%dx%d)\n"
            "  SPACE = one dream    P = autoplay (toggle)    ESC = quit\n",
            window_width, window_height);

    uint64_t iteration   = 0;
    double fps_window_t  = glfwGetTime();
    uint64_t fps_count   = 0;
    char fps_title[192];
    while (!glfwWindowShouldClose(window) && !g_quit_interactive.load()) {
        glfwPollEvents();

        // Generate when a SPACE press is pending, or continuously in autoplay.
        const bool want_gen = g_glfw_regen.exchange(false) || g_glfw_autoplay.load();
        if (want_gen) {
            iteration++;
            uint64_t seed = fixed_seed;
            if (seed != 0) {
                seed += iteration - 1;
            } else {
                std::random_device rd;
                seed = ((uint64_t)rd() << 32) ^ rd() ^ (uint64_t)time(nullptr) ^ (uint64_t)(uintptr_t)window;
            }
            fprintf(stderr, "sd-decoder: iteration %llu\n", (unsigned long long)iteration);
            build_random(seed);

            sd_image_t image = decode_only();
            if (image.width == 0 || image.data == nullptr) {
                fprintf(stderr, "sd-decoder: VAE decode failed\n");
                break;
            }
            upload_image_to_texture(image);
            free(image.data);
            fps_count++;
        }

        // Throughput sample: generations per second over a rolling ~2s window,
        // shown in the window title.
        const double now = glfwGetTime();
        if (now - fps_window_t >= 2.0) {
            const double rate = fps_count / (now - fps_window_t);
            snprintf(fps_title, sizeof(fps_title), "sd-decoder dream %dx%d | %.2f img/s%s",
                     window_width, window_height, rate,
                     g_glfw_autoplay.load() ? " [PLAY]" : "");
            glfwSetWindowTitle(window, fps_title);
            fps_window_t = now;
            fps_count    = 0;
        }

        glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(s_prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, s_tex);
        glBindVertexArray(s_quad_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glfwSwapBuffers(window);
    }

    fprintf(stderr, "sd-decoder: dream window closed after %llu iteration(s)\n",
            (unsigned long long)iteration);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
#endif  // SD_DECODER_HAS_GLFW

int main(int argc, const char** argv) {
    Options opt;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "sd-decoder: missing value for %s\n", arg.c_str());
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--vae") {
            const char* v = next();
            if (!v) return 1;
            opt.vae_path = v;
        } else if (arg == "--latent-file") {
            const char* v = next();
            if (!v) return 1;
            opt.latent_path = v;
        } else if (arg == "-o" || arg == "--output") {
            const char* v = next();
            if (!v) return 1;
            opt.output_path = v;
        } else if (arg == "-W" || arg == "--width") {
            const char* v = next();
            if (!v) return 1;
            opt.width = atoi(v);
        } else if (arg == "-H" || arg == "--height") {
            const char* v = next();
            if (!v) return 1;
            opt.height = atoi(v);
        } else if (arg == "--channels") {
            const char* v = next();
            if (!v) return 1;
            opt.channels_override = atoi(v);
        } else if (arg == "--model-family") {
            const char* v = next();
            if (!v) return 1;
            opt.model_family = sd_version_from_str(v);
            if (opt.model_family < 0) {
                fprintf(stderr, "sd-decoder: unknown model family '%s'\n", v);
                return 1;
            }
        } else if (arg == "--seed") {
            const char* v = next();
            if (!v) return 1;
            opt.seed = strtoull(v, nullptr, 10);
        } else if (arg == "--noise") {
            const char* v = next();
            if (!v) return 1;
            if (strcmp(v, "gaussian") == 0 || strcmp(v, "gauss") == 0 || strcmp(v, "normal") == 0) {
                opt.gaussian = true;
            } else if (strcmp(v, "uniform") == 0) {
                opt.gaussian = false;
            } else {
                fprintf(stderr, "sd-decoder: unknown noise mode '%s' (gaussian|uniform)\n", v);
                return 1;
            }
        } else if (arg == "--blur") {
            const char* v = next();
            if (!v) return 1;
            opt.blur = atoi(v);
            if (opt.blur < 0) opt.blur = 0;
        } else if (arg == "--amp") {
            const char* v = next();
            if (!v) return 1;
            opt.amp = atof(v);
            if (opt.amp < 0.0) opt.amp = 0.0;
        } else if (arg == "--loop") {
            opt.loop = true;
        } else if (arg == "--iterations") {
            const char* v = next();
            if (!v) return 1;
            opt.iterations = atoi(v);
            if (opt.iterations < 0) opt.iterations = 0;
        } else if (arg == "--n-threads") {
            const char* v = next();
            if (!v) return 1;
            opt.n_threads = atoi(v);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "sd-decoder: unknown argument %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (opt.vae_path.empty()) {
        fprintf(stderr, "sd-decoder: --vae is required\n");
        print_usage(argv[0]);
        return 1;
    }
    if (opt.latent_path.empty() && (opt.width <= 0 || opt.height <= 0)) {
        fprintf(stderr, "sd-decoder: invalid canvas size\n");
        return 1;
    }

    if (verbose) {
        sd_set_log_callback(log_cb, nullptr);
    }

    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.vae_path = opt.vae_path.c_str();
    if (opt.n_threads > 0) {
        ctx_params.n_threads = opt.n_threads;
    }

    // Resolve the model family: .lat v2 header > --model-family > default.
    std::vector<float> latent_data;
    std::vector<int64_t> latent_shape;
    int lat_model_version = -1;
    if (!opt.latent_path.empty()) {
        if (!read_lat_file(opt.latent_path, latent_data, latent_shape, lat_model_version)) {
            return 1;
        }
    }
    int version = opt.model_family;
    if (version < 0) {
        version = lat_model_version;
    }
    if (version < 0) {
        version = sd_version_from_str("Flux.2 klein");
        fprintf(stderr, "sd-decoder: no model family in .lat header; assuming '%s' "
                        "(pass --model-family to override)\n",
                version >= 0 ? sd_version_to_str(version) : "(unknown)");
    }
    if (version < 0) {
        fprintf(stderr, "sd-decoder: cannot resolve a model family; pass --model-family\n");
        return 1;
    }
    ctx_params.force_model_version = version;
    fprintf(stderr, "sd-decoder: model family: %s\n", sd_version_to_str(version));

    sd_ctx_t* sd_ctx = new_sd_ctx(&ctx_params);
    if (sd_ctx == nullptr) {
        fprintf(stderr, "sd-decoder: failed to create context (check --vae path and --model-family)\n");
        return 1;
    }

    // Decode the current latent_data. Returns the image (caller frees data).
    auto decode_only = [&]() -> sd_image_t {
        return sd_decode_latent(sd_ctx,
                                latent_data.data(),
                                latent_shape.data(),
                                static_cast<int>(latent_shape.size()));
    };

    // Decode latent_data and (over)write the output file. Returns true on success.
    auto decode_and_write = [&]() -> bool {
        sd_image_t image = decode_only();
        if (image.width == 0 || image.data == nullptr) {
            fprintf(stderr, "sd-decoder: VAE decode failed\n");
            return false;
        }
        const bool ok = write_image_to_file(opt.output_path,
                                            image.data,
                                            static_cast<int>(image.width),
                                            static_cast<int>(image.height),
                                            static_cast<int>(image.channel));
        fprintf(stderr, "sd-decoder: %s (%ux%u) -> %s\n",
                ok ? "decoded" : "FAILED to write",
                image.width, image.height, opt.output_path.c_str());
        free(image.data);
        return ok;
    };

    if (opt.latent_path.empty()) {
        // Random-blob mode: build a latent shaped like the canvas the VAE expects.
        int channels = opt.channels_override;
        if (channels <= 0) {
            channels = sd_ctx_get_vae_latent_channels(sd_ctx);
            if (channels > 0) {
                fprintf(stderr, "sd-decoder: detected %d latent channels from VAE weights\n", channels);
            } else {
                fprintf(stderr, "sd-decoder: could not detect latent channels; pass --channels N\n");
                free_sd_ctx(sd_ctx);
                return 1;
            }
        }
        // Latent layout is model-family dependent: FLUX.2's AE downsamples 16x
        // spatially (128 latent channels), classic KL AEs 8x. sd.cpp tensors are
        // [W/down, H/down, C] in shape() order.
        const std::string family = sd_version_to_str(version);
        const int64_t down = (family.find("Flux.2") == 0 || family.find("Flux.3") == 0) ? 16 : 8;
        const int64_t lw = opt.width / down;
        const int64_t lh = opt.height / down;
        if (lw <= 0 || lh <= 0) {
            fprintf(stderr, "sd-decoder: canvas too small\n");
            free_sd_ctx(sd_ctx);
            return 1;
        }
        latent_shape = {lw, lh, static_cast<int64_t>(channels), 1};  // trailing batch singleton, like the sampler's output
        latent_data.resize(static_cast<size_t>(lw * lh * channels));
        std::vector<float> scratch(latent_data.size());

        // Fill latent_data with shaped noise for a given seed (gaussian or uniform),
        // blur it, then renormalize each channel to unit std and scale by --amp.
        auto build_random = [&](uint64_t seed) {
            uint64_t rng = seed;
            auto next_u01 = [&]() {
                // xorshift64* — cheap deterministic RNG, uniform in [0, 1).
                rng ^= rng >> 12;
                rng ^= rng << 25;
                rng ^= rng >> 27;
                return (double)((rng * 2685821657736338717ull) >> 11) * (1.0 / 9007199254740992.0);
            };
            if (opt.gaussian) {
                // Box-Muller from the xorshift stream: N(0,1), deterministic per seed.
                for (float& v : latent_data) {
                    double u1 = next_u01();
                    if (u1 < 1e-12) u1 = 1e-12;  // keep ln(u1) finite
                    const double u2 = next_u01();
                    v               = (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
                }
            } else {
                for (float& v : latent_data) {
                    v = (float)(next_u01() * 2.0 - 1.0);
                }
            }
            // Spatial box blur: real latents are correlated over neighbouring cells; iid
            // noise has zero correlation and decodes to static. Blurring injects the
            // low-frequency structure. Layout: shape[0] (lw) is fastest, planes per channel.
            if (opt.blur > 0 && lw > 2 * opt.blur + 1 && lh > 2 * opt.blur + 1) {
                float* tmp = scratch.data();
                const int radius = opt.blur;
                for (int64_t c = 0; c < channels; c++) {
                    float* plane = latent_data.data() + c * (lw * lh);
                    float* dst   = tmp + c * (lw * lh);
                    for (int64_t y = 0; y < lh; y++) {
                        for (int64_t x = 0; x < lw; x++) {
                            const int64_t x0 = x - radius < 0 ? 0 : x - radius;
                            const int64_t x1 = x + radius >= lw ? lw - 1 : x + radius;
                            float sum        = 0.f;
                            for (int64_t k = x0; k <= x1; k++) sum += plane[y * lw + k];
                            dst[y * lw + x] = sum / (float)(x1 - x0 + 1);
                        }
                    }
                    for (int64_t y = 0; y < lh; y++) {
                        const int64_t y0 = y - radius < 0 ? 0 : y - radius;
                        const int64_t y1 = y + radius >= lh ? lh - 1 : y + radius;
                        for (int64_t x = 0; x < lw; x++) {
                            float sum = 0.f;
                            for (int64_t k = y0; k <= y1; k++) sum += dst[k * lw + x];
                            plane[y * lw + x] = sum / (float)(y1 - y0 + 1);
                        }
                    }
                }
            }
            // Renormalize each channel to zero mean / unit std, then scale by --amp.
            // Blur shrinks variance; without this the output would dim and grey out.
            if (opt.gaussian) {
                for (int64_t c = 0; c < channels; c++) {
                    float* plane = latent_data.data() + c * (lw * lh);
                    double mean  = 0.0, m2 = 0.0;
                    for (int64_t i = 0; i < lw * lh; i++) mean += plane[i];
                    mean /= (double)(lw * lh);
                    for (int64_t i = 0; i < lw * lh; i++) {
                        const double d = plane[i] - mean;
                        m2 += d * d;
                    }
                    const double std = sqrt(m2 / (double)(lw * lh)) + 1e-8;
                    const double s   = opt.amp / std;
                    for (int64_t i = 0; i < lw * lh; i++) plane[i] = (float)((plane[i] - mean) * s);
                }
            }
            fprintf(stderr, "sd-decoder: random latent seed=%llu shape=%lldx%lldx%lld noise=%s blur=%d amp=%.2f\n",
                    (unsigned long long)seed, (long long)lw, (long long)lh, (long long)channels,
                    opt.gaussian ? "gaussian" : "uniform", opt.blur, opt.amp);
        };

        auto entropy_seed = [&]() {
            // OS entropy per run (std::random_device on MSVC = RtlGenRandom).
            std::random_device rd;
            return ((uint64_t)rd() << 32) ^ rd() ^ (uint64_t)time(nullptr) ^ (uint64_t)(uintptr_t)&opt;
        };

        const bool interactive = opt.loop || opt.iterations > 0;
        if (!interactive) {
            build_random(opt.seed != 0 ? opt.seed : entropy_seed());
            if (!decode_and_write()) {
                free_sd_ctx(sd_ctx);
                return 1;
            }
        } else if (opt.iterations > 0) {
            // Bounded loop: N generations back-to-back, no keypress needed.
            const uint64_t fixed_seed = opt.seed;  // 0 = fresh entropy per iteration
            fprintf(stderr, "sd-decoder: generating %d dream images (overwriting '%s')\n",
                    opt.iterations, opt.output_path.c_str());
            for (uint64_t iteration = 1; iteration <= (uint64_t)opt.iterations; iteration++) {
                if (g_quit_interactive.load()) break;
                const uint64_t seed = fixed_seed != 0 ? fixed_seed + iteration - 1 : entropy_seed();
                fprintf(stderr, "sd-decoder: iteration %llu\n", (unsigned long long)iteration);
                build_random(seed);
                if (!decode_and_write()) {
                    return 1;  // decode failed
                }
            }
            fprintf(stderr, "sd-decoder: dream loop ended (%d iterations)\n", opt.iterations);
        } else {
            // Interactive --loop: VAE stays loaded; a keypress generates a new dream
            // image overwriting the output file.
            uint64_t fixed_seed = opt.seed;  // 0 = fresh entropy per keypress
            g_quit_interactive.store(false);
#if SD_DECODER_HAS_GLFW
            const int ret = glfw_dream_loop(static_cast<int>(opt.width),
                                            static_cast<int>(opt.height),
                                            fixed_seed,
                                            build_random,
                                            decode_only);
            if (ret != 0) {
                free_sd_ctx(sd_ctx);
                return ret;
            }
#else
            // Console fallback (build without GLFW): any stdin key = next dream.
            uint64_t iteration = 0;
#ifndef _WIN32
            struct sigaction sa;
            memset(&sa, 0, sizeof(sa));
            sa.sa_handler = [](int) { g_quit_interactive.store(true); };
            sigaction(SIGINT, &sa, nullptr);
#else
            SetConsoleCtrlHandler([](DWORD type) -> BOOL {
                if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
                    g_quit_interactive.store(true);
                    return TRUE;  // swallow -> we exit via the flag
                }
                return FALSE;
            }, TRUE);
#endif
            fprintf(stderr,
                    "sd-decoder: dream mode (console) - press any key for a new dream image\n"
                    "(overwrites '%s'), Ctrl-C to quit.\n",
                    opt.output_path.c_str());
            while (!g_quit_interactive.load()) {
#ifdef _WIN32
                if (!_kbhit()) {
                    Sleep(40);
                    continue;
                }
                (void)_getch();  // consume the key
#else
                fd_set rfds;
                FD_ZERO(&rfds);
                FD_SET(0, &rfds);
                struct timeval tv = {0, 40000};
                if (select(1, &rfds, nullptr, nullptr, &tv) <= 0) continue;
                (void)getchar();
#endif
                iteration++;
                const uint64_t seed = fixed_seed != 0 ? fixed_seed + iteration - 1 : entropy_seed();
                fprintf(stderr, "sd-decoder: iteration %llu\n", (unsigned long long)iteration);
                build_random(seed);
                if (!decode_and_write()) {
                    break;  // decode failed - leave the loop rather than spin
                }
            }
            fprintf(stderr, "sd-decoder: dream loop ended (iteration %llu)\n", (unsigned long long)iteration);
#endif  // SD_DECODER_HAS_GLFW
        }
    } else {
        fprintf(stderr, "sd-decoder: loaded %zu-element latent from '%s'\n",
                latent_data.size(), opt.latent_path.c_str());
        if (opt.loop || opt.iterations > 0) {
            fprintf(stderr, "sd-decoder: --loop/--iterations only apply to random (no --latent-file) mode\n");
        }
        if (!decode_and_write()) {
            free_sd_ctx(sd_ctx);
            return 1;
        }
    }

    free_sd_ctx(sd_ctx);
    return 0;
}
