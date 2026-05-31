package main

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/tmc/mlx-go-ane/aneenv"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/runtime/decodeplane"
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
		return decodeplane.DecodePlaneOff, nil
	case "auto", "qwen35", "qwen3.5", "qwen3_5":
		return decodeplane.DecodePlaneAuto, nil
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
	if *aneDecodePlane == decodeplane.DecodePlaneOff {
		return model
	}
	wrapped, err := decodeplane.Wrap(context.Background(), model, decodeplane.DecodePlaneOptions{
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
