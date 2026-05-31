package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/tmc/mlx-go-lm/mlxlm/hfcache"
)

// resolveModelPath resolves a model path to a local directory,
// downloading from HuggingFace if it looks like a model ID.
func resolveModelPath(path string) (string, error) {
	if _, err := os.Stat(path); err == nil {
		return filepath.Abs(path)
	}
	if strings.Contains(path, "/") {
		cache, err := hfcache.NewHFCacheWithOptions(*autoDownload)
		if err != nil {
			return "", fmt.Errorf("failed to initialize cache: %w", err)
		}
		return cache.GetModelPath(context.Background(), path)
	}
	return "", fmt.Errorf("model path not found: %s", path)
}
