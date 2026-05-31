package main

import (
	"fmt"
	"time"

	"github.com/tmc/mlx-go-lm/mlxlm/llm/models"
	"github.com/tmc/mlx-go-lm/mlxlm/llm/runtime/decodeplane"
	"github.com/tmc/mlx-go/mlx"
)

type generationStats struct {
	PrefillTPS             float64
	GenerationTPS          float64
	GenerationOnlyDuration time.Duration
}

func computeGenerationStats(
	promptTokens int,
	generatedTokenCount int,
	prefillDuration time.Duration,
	genDuration time.Duration,
) generationStats {
	st := generationStats{}
	if prefillDuration.Seconds() > 0 {
		st.PrefillTPS = float64(promptTokens) / prefillDuration.Seconds()
	}
	st.GenerationOnlyDuration = genDuration - prefillDuration
	if generatedTokenCount > 0 && st.GenerationOnlyDuration.Seconds() > 0 {
		st.GenerationTPS = float64(generatedTokenCount) / st.GenerationOnlyDuration.Seconds()
	}
	return st
}

func printStatistics(
	quietMode bool,
	promptTokens int,
	generatedTokens []int32,
	prefillDuration time.Duration,
	genDuration time.Duration,
) {
	if quietMode {
		return
	}

	fmt.Println()

	peakMemoryGB := 0.0
	activeMemoryGB := 0.0
	cacheMemoryGB := 0.0

	peakBytes := mlx.GetPeakMemory()
	peakMemoryGB = float64(peakBytes) / (1024 * 1024 * 1024)

	if *memoryStats {
		activeBytes := mlx.GetActiveMemory()
		activeMemoryGB = float64(activeBytes) / (1024 * 1024 * 1024)
		cacheBytes := mlx.GetCacheMemory()
		cacheMemoryGB = float64(cacheBytes) / (1024 * 1024 * 1024)
	}

	stats := computeGenerationStats(promptTokens, len(generatedTokens), prefillDuration, genDuration)

	fmt.Printf("Prefill: %d tokens, %.3f tokens-per-sec (%.3fs)\n", promptTokens, stats.PrefillTPS, prefillDuration.Seconds())
	fmt.Printf("Generation: %d tokens, %.3f tokens-per-sec (%.3fs)\n", len(generatedTokens), stats.GenerationTPS, stats.GenerationOnlyDuration.Seconds())
	fmt.Printf("Peak memory: %.3f GB\n", peakMemoryGB)

	if *memoryStats {
		fmt.Printf("Active memory: %.3f GB\n", activeMemoryGB)
		fmt.Printf("Cache memory: %.3f GB\n", cacheMemoryGB)
	}
}

func printANEDecodePlaneStatistics(quietMode bool, model models.LanguageModel) {
	if quietMode || model == nil {
		return
	}
	reporter, ok := model.(decodeplane.DecodePlaneStatsReporter)
	if !ok {
		return
	}
	st := reporter.DecodePlaneStats()
	if !st.Enabled {
		return
	}

	fmt.Printf(
		"ANE decode plane: stage-builds=%d (dense=%d shared=%d expert=%d, failures=%d), init=%.3fs\n",
		st.StageBuilds,
		st.DenseStageBuilds,
		st.SharedStageBuilds,
		st.ExpertStageBuilds,
		st.StageBuildFailures,
		st.StageBuildDuration.Seconds(),
	)
	if st.StageBuilds > 0 {
		fmt.Printf(
			"ANE decode plane: artifact-cache=%d hit / %d miss, artifact=%.3fs, compile=%.3fs, map=%.3fs\n",
			st.ArtifactCacheHits,
			st.ArtifactCacheMisses,
			st.ArtifactReady.Seconds(),
			st.CompileLoad.Seconds(),
			st.MapDuration.Seconds(),
		)
	}
	fmt.Printf(
		"ANE decode plane: steps=%d (dense=%d moe=%d, sync=%d), stage-calls=%d (sync=%d), prepare=%.3fs, ane=%.3fs, output=%.3fs\n",
		st.DecodeSteps,
		st.DenseSteps,
		st.MoESteps,
		st.SynchronizedDecodeSteps,
		st.StageCalls,
		st.SynchronizedStageCalls,
		st.PrepareDuration.Seconds(),
		st.ANEDuration.Seconds(),
		st.OutputDuration.Seconds(),
	)
	if st.StageCalls > 0 {
		fmt.Printf(
			"ANE decode plane: input avg alias=%.3fms, avg eval=%.3fms, avg copy=%.3fms, avg finalize=%.3fms\n",
			durationAverageMS(st.InputAliasDuration, st.StageCalls),
			durationAverageMS(st.InputEvalDuration, st.StageCalls),
			durationAverageMS(st.InputCopyDuration, st.StageCalls),
			durationAverageMS(st.StreamFinalizeDuration, max(1, st.DecodeSteps)),
		)
	}
	if st.DenseSteps > 0 {
		fmt.Printf(
			"ANE decode plane: dense avg prepare=%.3fms, avg ane=%.3fms, avg output=%.3fms\n",
			durationAverageMS(st.DensePrepareDuration, st.DenseSteps),
			durationAverageMS(st.DenseANEDuration, st.DenseSteps),
			durationAverageMS(st.DenseOutputDuration, st.DenseSteps),
		)
	}
	if st.MoESteps > 0 {
		fmt.Printf(
			"ANE decode plane: moe avg prepare=%.3fms, avg ane=%.3fms, avg output=%.3fms, avg router=%.3fms, avg combine=%.3fms\n",
			durationAverageMS(st.MoEPrepareDuration, st.MoESteps),
			durationAverageMS(st.MoEANEDuration, st.MoESteps),
			durationAverageMS(st.MoEOutputDuration, st.MoESteps),
			durationAverageMS(st.MoERouterDuration, st.MoESteps),
			durationAverageMS(st.MoECombineDuration, st.MoESteps),
		)
	}
	if st.DecodeSteps > 0 {
		fmt.Printf(
			"ANE decode plane: avg-step=%.3fms, avg-ane-stage=%.3fms, sync-step-rate=%.1f%%, sync-stage-rate=%.1f%%, host-input-fallbacks=%d\n",
			durationAverageMS(st.TotalStepDuration, st.DecodeSteps),
			durationAverageMS(st.ANEDuration, st.StageCalls),
			ratioPercent(st.SynchronizedDecodeSteps, st.DecodeSteps),
			ratioPercent(st.SynchronizedStageCalls, st.StageCalls),
			st.EffectiveHostInputFallbacks(),
		)
	} else {
		fmt.Printf(
			"ANE decode plane: sync-stage-rate=%.1f%%, host-input-fallbacks=%d\n",
			ratioPercent(st.SynchronizedStageCalls, st.StageCalls),
			st.EffectiveHostInputFallbacks(),
		)
	}
	if st.Disabled {
		fmt.Printf("ANE decode plane: disabled after error: %s\n", st.DisableReason)
	}
}

func durationAverageMS(total time.Duration, count int) float64 {
	if count <= 0 {
		return 0
	}
	return float64(total) / float64(time.Millisecond) / float64(count)
}

func ratioPercent(num, den int) float64 {
	if den <= 0 {
		return 0
	}
	return 100 * float64(num) / float64(den)
}
