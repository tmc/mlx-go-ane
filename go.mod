module github.com/tmc/mlx-go-ane

go 1.26

require (
	github.com/tmc/mlx-go v0.0.0
	github.com/tmc/mlx-go-lm v0.0.0
)

require (
	github.com/ebitengine/purego v0.10.0 // indirect
	github.com/tmc/modelir v0.1.2-0.20260517090425-24c01509645e // indirect
	golang.org/x/image v0.36.0 // indirect
)

replace github.com/tmc/mlx-go => ../mlx-go

replace github.com/tmc/mlx-go-lm => ../mlx-go-worktrees/mlx-go-lm

replace github.com/ebitengine/purego => github.com/tmc/purego v0.10.0-alpha.2.0.20260207193206-ff6e796b10b1
