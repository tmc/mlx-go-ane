package main

import (
	"testing"

	"github.com/tmc/mlx-go-lm/mlxlm/kvcache"
)

func TestBuildCacheConfigInplaceAliasUsesPrealloc(t *testing.T) {
	resetCacheFlags(t)
	*inplaceCache = true

	config := buildCacheConfig()
	if config.Type != kvcache.TypePrealloc {
		t.Fatalf("config.Type = %v, want %v", config.Type, kvcache.TypePrealloc)
	}
}

func TestBuildCacheConfigQuantized(t *testing.T) {
	resetCacheFlags(t)
	*preallocCache = true
	*kvBits = 4
	*kvGroupSize = 32
	*kvSize = 128
	*quantizedKVStart = 64

	config := buildCacheConfig()
	if config.Type != kvcache.TypeQuantized {
		t.Fatalf("config.Type = %v, want %v", config.Type, kvcache.TypeQuantized)
	}
	if config.KVBits != 4 {
		t.Fatalf("config.KVBits = %d, want 4", config.KVBits)
	}
	if config.KVGroupSize != 32 {
		t.Fatalf("config.KVGroupSize = %d, want 32", config.KVGroupSize)
	}
	if config.PreallocStep != 128 {
		t.Fatalf("config.PreallocStep = %d, want 128", config.PreallocStep)
	}
}

func TestBuildCacheConfigRotating(t *testing.T) {
	resetCacheFlags(t)
	*rotatingCache = true
	*maxKVSize = 1024
	*keepTokens = 16

	config := buildCacheConfig()
	if config.Type != kvcache.TypeRotating {
		t.Fatalf("config.Type = %v, want %v", config.Type, kvcache.TypeRotating)
	}
	if config.RotatingMaxSize != 1024 {
		t.Fatalf("config.RotatingMaxSize = %d, want 1024", config.RotatingMaxSize)
	}
	if config.RotatingKeep != 16 {
		t.Fatalf("config.RotatingKeep = %d, want 16", config.RotatingKeep)
	}
}

func resetCacheFlags(t *testing.T) {
	t.Helper()

	oldInplaceCache := *inplaceCache
	oldRotatingCache := *rotatingCache
	oldPreallocCache := *preallocCache
	oldMaxKVSize := *maxKVSize
	oldKeepTokens := *keepTokens
	oldKVBits := *kvBits
	oldKVGroupSize := *kvGroupSize
	oldQuantizedKVStart := *quantizedKVStart
	oldKVSize := *kvSize

	*inplaceCache = false
	*rotatingCache = false
	*preallocCache = false
	*maxKVSize = 0
	*keepTokens = 0
	*kvBits = 0
	*kvGroupSize = 64
	*quantizedKVStart = 0
	*kvSize = 0

	t.Cleanup(func() {
		*inplaceCache = oldInplaceCache
		*rotatingCache = oldRotatingCache
		*preallocCache = oldPreallocCache
		*maxKVSize = oldMaxKVSize
		*keepTokens = oldKeepTokens
		*kvBits = oldKVBits
		*kvGroupSize = oldKVGroupSize
		*quantizedKVStart = oldQuantizedKVStart
		*kvSize = oldKVSize
	})
}
