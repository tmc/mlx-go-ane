package main

import "flag"

var (
	// Model flags
	modelPath    = flag.String("model", "mlx-community/Qwen2.5-3B-Instruct-4bit", "Path to model directory or HuggingFace model ID")
	autoDownload = flag.Bool("auto-download", true, "Automatically download model from HuggingFace if not cached")

	// Generation flags
	systemPrompt      = flag.String("system", "", "The system prompt")
	systemPromptShort = flag.String("s", "", "The system prompt (short form)")
	prompt            = flag.String("prompt", "", "The message to be processed by the model. Use @path to load from files")
	promptShort       = flag.String("p", "", "The message to be processed by the model (short form)")
	maxTokens         = flag.Int("max-tokens", 100, "Maximum number of tokens to generate")
	maxTokensShort    = flag.Int("m", 100, "Maximum number of tokens to generate (short form)")
	temperature       = flag.Float64("temperature", 0.0, "The sampling temperature")
	temperatureShort  = flag.Float64("t", 0.0, "The sampling temperature (short form)")
	topP              = flag.Float64("top-p", 1.0, "The top p sampling")
	topPShort         = flag.Float64("tp", 1.0, "The top p sampling (short form)")
	minP              = flag.Float64("min-p", 0.0, "The min p sampling")
	minPShort         = flag.Float64("mp", 0.0, "The min p sampling (short form)")
	topK              = flag.Int("top-k", 0, "The top k sampling")
	topKShort         = flag.Int("tk", 0, "The top k sampling (short form)")
	extraEOSToken     = flag.String("extra-eos-token", "", "Additional end-of-sequence token to stop generation")
	seed              = flag.Int("seed", 0, "The PRNG seed")
	quiet             = flag.Bool("quiet", false, "If true only print the generated output")
	quietShort        = flag.Bool("q", false, "If true only print the generated output (short form)")

	// KV cache flags
	inplaceCache     = flag.Bool("inplace-cache", false, "Use in-place KV cache via C++ FFI")
	rotatingCache    = flag.Bool("rotating-cache", false, "Use rotating KV cache (ring buffer with fixed max size)")
	preallocCache    = flag.Bool("prealloc-cache", false, "Use pre-allocated KV cache")
	maxKVSize        = flag.Int("max-kv-size", 0, "Maximum KV cache size for rotating cache")
	keepTokens       = flag.Int("keep", 0, "Number of prefix tokens to preserve in rotating cache")
	kvBits           = flag.Int("kv-bits", 0, "Number of bits for KV cache quantization (0 = no quantization)")
	kvGroupSize      = flag.Int("kv-group-size", 64, "Group size for KV cache quantization")
	quantizedKVStart = flag.Int("quantized-kv-start", 0, "Step to begin using quantized KV cache when kv-bits is set")
	kvSize           = flag.Int("kv-size", 0, "Size of the KV cache (0 = model max)")

	// Display/debug flags
	verbose     = flag.Bool("verbose", false, "Verbose output")
	debug       = flag.Bool("debug", false, "Debug output")
	memoryStats = flag.Bool("memory-stats", false, "Show detailed memory statistics")
	warmup      = flag.Bool("warmup", false, "Run warmup pass before generation")

	// Chat template flag
	useChatTemplate   = flag.Bool("use-chat-template", true, "Wrap prompt in model-specific chat template")
	noUseChatTemplate = flag.Bool("no-chat-template", false, "Disable chat template, send prompt as raw text")

	// Profiling flags
	cpuProfile = flag.String("cpuprofile", "", "Write CPU profile to file")
	memProfile = flag.String("memprofile", "", "Write memory profile to file")
)

// getFloat64Flag returns the long-form value unless the short form was explicitly changed.
func getFloat64Flag(long, short *float64, defaultVal float64) float64 {
	if *long != defaultVal {
		return *long
	}
	return *short
}
