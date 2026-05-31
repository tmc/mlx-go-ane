package main

import (
	"log/slog"

	"github.com/tmc/mlx-go-lm/mlxlm/kvcache"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go/mlx"
)

// buildCacheConfig creates a kvcache.Config based on CLI flags.
func buildCacheConfig() *kvcache.Config {
	config := kvcache.DefaultConfig()

	switch {
	case *inplaceCache && mlx.HasKVCacheInplace():
		config.Type = kvcache.TypePrealloc
		slog.Info("Using in-place KV cache via C++ FFI")
	case *rotatingCache:
		config.Type = kvcache.TypeRotating
		if *maxKVSize > 0 {
			config.RotatingMaxSize = *maxKVSize
		} else if *kvSize > 0 {
			config.RotatingMaxSize = *kvSize
		} else {
			config.RotatingMaxSize = 2048
			slog.Info("Rotating cache enabled but --max-kv-size not set. Defaulting to 2048.")
		}
		config.RotatingKeep = *keepTokens
		slog.Info("Using rotating KV cache", "maxSize", config.RotatingMaxSize, "keep", config.RotatingKeep)
	case *preallocCache:
		config.Type = kvcache.TypePrealloc
		slog.Info("Using pre-allocated KV cache")
	default:
		config.Type = kvcache.TypeDefault
	}

	if *kvSize > 0 {
		config.PreallocStep = *kvSize
	}

	return config
}

// createMultiLayerCache creates a MultiLayerCache with the given configuration.
func createMultiLayerCache(numLayers int, config *kvcache.Config) *models.MultiLayerCache {
	caches := make([]kvcache.Cache, numLayers)
	for i := range caches {
		cache, err := kvcache.New(config)
		if err != nil {
			slog.Warn("Failed to create cache, falling back to concat", "error", err, "layer", i)
			cache, _ = kvcache.NewConcat()
		}
		caches[i] = cache
	}
	return models.NewMultiLayerCacheFromList(caches)
}
