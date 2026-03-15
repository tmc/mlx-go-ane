// Package aneenv reads MLXGO_ANE_* environment variables with type-safe defaults.
package aneenv

import (
	"os"
	"strconv"
)

// String returns the environment value or the default.
func String(key, def string) string {
	v, ok := os.LookupEnv(key)
	if !ok {
		v = def
	}
	return v
}

// Int returns the integer environment value or the default.
func Int(key string, def int) int {
	v := def
	if raw, ok := os.LookupEnv(key); ok {
		if n, err := strconv.Atoi(raw); err == nil {
			v = n
		}
	}
	return v
}

// Bool returns the boolean environment value or the default.
func Bool(key string, def bool) bool {
	v := def
	if raw, ok := os.LookupEnv(key); ok {
		if b, err := strconv.ParseBool(raw); err == nil {
			v = b
		}
	}
	return v
}

// IsSet reports whether the environment variable is set.
func IsSet(key string) bool {
	_, ok := os.LookupEnv(key)
	return ok
}
