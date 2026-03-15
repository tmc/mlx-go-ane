package main

import (
	"fmt"
	"log/slog"
	"strings"

	"github.com/tmc/mlx-go-lm/mlxlm/aneenv"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/runtime/anedecode"
	"github.com/tmc/mlx-go/mlx"
)

var (
	aneDecodePlane = new(string)
	aneDecodeCache = new(string)
)

func loadANEEnv() {
	// Default to "qwen35" (enabled) — the key difference from mlx-lm-generate.
	*aneDecodePlane = aneenv.String("MLXGO_ANE_DECODE_PLANE", "qwen35")
	*aneDecodeCache = aneenv.String("MLXGO_ANE_DECODE_CACHE", "")
}

func normalizeANEDecodePlaneMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "off", "none", "disabled":
		return anedecode.ANEDecodePlaneOff, nil
	case "qwen35", "qwen3.5", "qwen3_5":
		return anedecode.ANEDecodePlaneQwen35, nil
	default:
		return "", fmt.Errorf("unsupported ANE decode plane %q", raw)
	}
}

func validateANEDecodePlaneFlags() error {
	mode, err := normalizeANEDecodePlaneMode(*aneDecodePlane)
	if err != nil {
		return err
	}
	*aneDecodePlane = mode
	return nil
}

func wrapANEDecodePlane(model models.LanguageModel, modelPath string) models.LanguageModel {
	if *aneDecodePlane == anedecode.ANEDecodePlaneOff {
		return model
	}
	wrapped, err := anedecode.Wrap(model, anedecode.ANEDecodePlaneOptions{
		Mode:      *aneDecodePlane,
		ModelPath: modelPath,
		CacheDir:  *aneDecodeCache,
		Warn: func(format string, args ...any) {
			slog.Warn(fmt.Sprintf(format, args...))
		},
	})
	if err != nil {
		slog.Warn("ANE decode plane unavailable; using stock MLX path", "error", err)
		return model
	}
	if err := mlx.SetCompileMode(mlx.CompileModeDisabled); err != nil {
		slog.Warn("failed to disable MLX compile mode for ANE decode plane", "error", err)
	} else {
		slog.Info("ANE decode plane enabled", "mode", *aneDecodePlane, "cache", *aneDecodeCache)
	}
	return wrapped
}
