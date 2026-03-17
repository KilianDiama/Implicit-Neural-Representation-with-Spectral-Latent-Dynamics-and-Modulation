#include <torch/torch.h>
#include <complex>

// ==========================================================
// 🔹 ModulatedLayer V3 (Zero-Init Style / Residual Ready)
// ==========================================================
struct ModulatedLayer : torch::nn::Module {
    torch::nn::Linear layer{nullptr}, style_proj{nullptr};
    int out_features;

    ModulatedLayer(int in_features, int out_features_, int latent_dim)
        : out_features(out_features_) {
        
        layer = register_module("layer", torch::nn::Linear(in_features, out_features));
        style_proj = register_module("style_proj", torch::nn::Linear(latent_dim, 2 * out_features));

        // Kaiming pour le chemin principal
        torch::nn::init::kaiming_normal_(layer->weight, 0, torch::kFanIn, torch::kSiLU);
        
        // Zero-init pour la modulation : le réseau commence en mode "Pass-through" stable
        torch::nn::init::zeros_(style_proj->weight);
        torch::nn::init::zeros_(style_proj->bias);
        style_proj->bias.slice(0, 0, out_features).fill_(1.0); 
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& z) {
        auto h = layer->forward(x);
        auto style = style_proj->forward(z).unsqueeze(1); // Broadcast-ready [B, 1, 2C]
        
        auto chunks = style.chunk(2, -1);
        return h * chunks[0] + chunks[1];
    }
};

// ==========================================================
// 🔹 UltimateLevel5Engine - Perfected Version
// ==========================================================
class UltimateEngineV12 : public torch::nn::Module {
private:
    int latent_dim;
    int steps;

    torch::nn::Sequential encoder{nullptr};
    torch::nn::LayerNorm latent_norm{nullptr};

    // Dynamics Core
    torch::nn::Linear spectral_gate{nullptr};
    torch::Tensor k_angles, k_magnitudes, inertia_gate;

    // Decoding
    torch::nn::Linear dec_proj_coords{nullptr};
    ModulatedLayer mod1{nullptr}, mod2{nullptr};
    torch::nn::Linear dec_final{nullptr};

public:
    UltimateEngineV12(int latent_dim_ = 512, int steps_ = 4)
        : latent_dim(latent_dim_), steps(steps_) {

        // 1. ENCODER (Deep ResNet-style block)
        encoder = register_module("encoder", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(2).padding(1)),
            torch::nn::GroupNorm(8, 64), torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)),
            torch::nn::GroupNorm(8, 128), torch::nn::SiLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, latent_dim, 3).stride(2).padding(1)),
            torch::nn::AdaptiveAvgPool2d({1, 1}),
            torch::nn::Flatten()
        ));

        latent_norm = register_module("latent_norm", torch::nn::LayerNorm(latent_dim));

        // 2. SPECTRAL CORE (Orthogonal Init)
        spectral_gate = register_module("spectral_gate", torch::nn::Linear(latent_dim, latent_dim));
        torch::nn::init::orthogonal_(spectral_gate->weight, 0.01);

        k_angles = register_parameter("k_angles", torch::randn({latent_dim / 2}) * 0.02);
        k_magnitudes = register_parameter("k_magnitudes", torch::zeros({latent_dim / 2}));
        inertia_gate = register_parameter("inertia_gate", torch::tensor(0.0)); // Sigmoid(0) = 0.5

        // 3. DECODER
        dec_proj_coords = register_module("dec_proj_coords", torch::nn::Linear(63, 256));
        mod1 = register_module("mod1", ModulatedLayer(256, 512, latent_dim));
        mod2 = register_module("mod2", ModulatedLayer(512, 256, latent_dim));
        dec_final = register_module("dec_final", torch::nn::Linear(256, 4));
    }

    torch::Tensor step(const torch::Tensor& z) {
        int half = latent_dim / 2;
        auto spec = spectral_gate->forward(z);

        // Splitting
        auto d_theta = spec.narrow(1, 0, half);
        auto d_mag   = spec.narrow(1, half, half);

        // Transformation complexe stable
        auto z_complex = torch::view_as_complex(z.view({z.size(0), half, 2}));
        
        // Soft-constraints pour empêcher l'explosion
        auto theta = k_angles + 0.1 * torch::tanh(d_theta);
        auto mag   = torch::exp(0.01 * (k_magnitudes + d_mag)); // Croissance contrôlée

        auto rotation = torch::polar(mag, theta);
        auto z_next = torch::view_as_real(z_complex * rotation).view({z.size(0), latent_dim});

        // Residual gate (Learnable)
        auto alpha = torch::sigmoid(inertia_gate);
        return torch::lerp(z, z_next, alpha);
    }

    torch::Tensor forward(torch::Tensor img, torch::Tensor coords) {
        // Feature extraction
        auto z = latent_norm->forward(encoder->forward(img));

        // ODE-like evolution
        for (int i = 0; i < steps; ++i) {
            z = step(z);
        }

        // Modulated Synthesis
        auto h = torch::nn::functional::silu(dec_proj_coords(coords));
        h = torch::nn::functional::silu(mod1->forward(h, z));
        h = torch::nn::functional::silu(mod2->forward(h, z));

        return dec_final->forward(h);
    }
};
