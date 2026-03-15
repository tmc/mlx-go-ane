package main

import (
	"context"
	"iter"
	"log/slog"

	"github.com/tmc/mlx-go-lm/mlxlm"
	"github.com/tmc/mlx-go-lm/mlxlm/decode"
	"github.com/tmc/mlx-go-lm/mlxlm/models"
	"github.com/tmc/mlx-go/mlx"
)

// Generation represents output from token generation.
type Generation struct {
	Chunk *string                 // Text chunk (decoded token)
	Info  *GenerateCompletionInfo // Completion metadata
}

// GenerateCompletionInfo contains generation statistics.
type GenerateCompletionInfo struct {
	PromptTokenCount     int
	GenerationTokenCount int
}

func newStreamingDetokenizer(tokenizer interface{ Decode([]int32) (string, error) }) mlxlm.StreamingDetokenizer {
	if tokenizer == nil {
		return nil
	}
	fullTokenizer, ok := tokenizer.(mlxlm.Tokenizer)
	if !ok {
		return nil
	}
	modelPath := ""
	if pat, ok := tokenizer.(interface{ ModelPath() string }); ok {
		modelPath = pat.ModelPath()
	}
	return mlxlm.NewStreamingDetokenizer(fullTokenizer, modelPath)
}

// Generate produces a sequence of Generation items using the ANE decode path.
func Generate(
	forwardPass decode.ForwardFunc,
	input *mlx.Array,
	cache models.Cache,
	temperature float64,
	topP float64,
	minP float64,
	topK int,
	eosTokens []int32,
	maxTokens int,
	tokenizer interface{ Decode([]int32) (string, error) },
	randomState *decode.RandomState,
) iter.Seq2[Generation, error] {

	return func(yield func(Generation, error) bool) {
		promptTokenCount := input.Size()

		opts := decode.Options{
			Temperature:     temperature,
			TopP:            topP,
			MinP:            minP,
			TopK:            topK,
			MaxTokens:       maxTokens,
			EOSTokens:       eosTokens,
			SamplingStrategy: "lazy",
			EagerPrefill:    false,
			UseStridedCache: false,
		}

		slog.Debug("Generate: creating TokenIterator", "strategy", "lazy")
		iterator, err := decode.NewTokenIterator(context.Background(), forwardPass, input, cache, opts)
		if err != nil {
			yield(Generation{}, err)
			return
		}

		if randomState != nil {
			iterator.SetRandomState(randomState)
		}

		var tokens []int32
		detok := newStreamingDetokenizer(tokenizer)
		streamedBytes := 0
		lastDecoded := 0

		for token, err := range iterator.Tokens() {
			if err != nil {
				yield(Generation{}, err)
				return
			}

			tokens = append(tokens, token)

			if detok != nil {
				chunk, err := detok.Append(token)
				if err == nil && chunk != "" {
					streamedBytes += len(chunk)
					if !yield(Generation{Chunk: &chunk}, nil) {
						return
					}
				}
			} else if tokenizer != nil {
				fullText, err := tokenizer.Decode(tokens)
				if err == nil {
					if len(fullText) > lastDecoded {
						chunk := fullText[lastDecoded:]
						lastDecoded = len(fullText)
						if !yield(Generation{Chunk: &chunk}, nil) {
							return
						}
					}
				}
			}
		}
		if detok != nil {
			fullText, err := detok.FullText()
			if err == nil && len(fullText) > streamedBytes {
				chunk := fullText[streamedBytes:]
				if chunk != "" {
					if !yield(Generation{Chunk: &chunk}, nil) {
						return
					}
				}
			}
		}

		info := GenerateCompletionInfo{
			PromptTokenCount:     promptTokenCount,
			GenerationTokenCount: len(tokens),
		}
		yield(Generation{Info: &info}, nil)
	}
}
