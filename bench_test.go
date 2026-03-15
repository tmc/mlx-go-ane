package mlxgoane

import (
	"fmt"
	"os"
	"runtime"
	"testing"

	"github.com/tmc/mlx-go/mlx"
)

var (
	testEngine       *Engine
	testPromptTokens []int32
)

func TestMain(m *testing.M) {
	modelID := os.Getenv("MODEL")
	if modelID == "" {
		modelID = DefaultModel
	}

	fmt.Fprintf(os.Stderr, "loading model %s...\n", modelID)
	var err error
	testEngine, err = setupEngine(modelID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "setup engine: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "model loaded from %s\n", testEngine.ModelPath)

	prompt := os.Getenv("PROMPT")
	if prompt == "" {
		prompt = DefaultPrompt
	}
	testPromptTokens, err = testEngine.encodePrompt(prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "encode prompt: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "prompt tokens: %d\n", len(testPromptTokens))

	if WarmupEnabled {
		fmt.Fprintf(os.Stderr, "warming up...\n")
		testEngine.warmup()
	}
	fmt.Fprintf(os.Stderr, "ready\n")

	code := m.Run()
	os.Exit(code)
}

// BenchmarkGenerate measures end-to-end generation (prefill + decode).
// Reports: tok/s, prefill_ms, gen_ms, peak_mem_gb.
func BenchmarkGenerate(b *testing.B) {
	var lastRes GenerateResult
	for b.Loop() {
		res, err := testEngine.generateN(testPromptTokens, GenerateTokens)
		if err != nil {
			b.Fatal(err)
		}
		lastRes = res
	}
	b.StopTimer()

	b.ReportMetric(lastRes.TokPerSec(), "tok/s")
	b.ReportMetric(float64(lastRes.PrefillDuration.Milliseconds()), "prefill_ms")
	b.ReportMetric(float64(lastRes.GenerateDuration.Milliseconds()), "gen_ms")

	peakMemGB := 0.0
	if peakBytes, err := mlx.GetPeakMemory(); err == nil {
		peakMemGB = float64(peakBytes) / (1024 * 1024 * 1024)
	} else {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		peakMemGB = float64(m.Sys) / (1024 * 1024 * 1024)
	}
	b.ReportMetric(peakMemGB, "peak_mem_gb")
}

// BenchmarkPrefill measures prefill-only throughput (prompt encoding + first token).
// Reports: prompt_tok/s.
func BenchmarkPrefill(b *testing.B) {
	for b.Loop() {
		res, err := testEngine.generateN(testPromptTokens, 1)
		if err != nil {
			b.Fatal(err)
		}
		b.ReportMetric(res.PrefillTokPerSec(), "prompt_tok/s")
	}
}

// BenchmarkDecode measures decode-only throughput (per-token generation after prefill).
// Reports: decode_tok/s.
func BenchmarkDecode(b *testing.B) {
	var lastRes GenerateResult
	for b.Loop() {
		res, err := testEngine.generateN(testPromptTokens, GenerateTokens)
		if err != nil {
			b.Fatal(err)
		}
		lastRes = res
	}
	b.StopTimer()

	b.ReportMetric(lastRes.DecodeTokPerSec(), "decode_tok/s")
}
