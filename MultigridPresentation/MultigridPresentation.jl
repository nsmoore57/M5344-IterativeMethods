### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 0c8ddd80-0764-11eb-3076-f57c55c75d89
begin 
	import Pkg
	Pkg.activate(".")
	using FFTW
	using AbstractFFTs
	using Plots
	using IterativeSolvers
	using SparseArrays
	using DSP
end

# ╔═╡ 994548e0-0768-11eb-12cd-21726bbca62f
begin
	N = 100
	xs = 1/(N+1)
	x0 = 0
	xmax = 1.0
end

# ╔═╡ 3e1df2d0-076a-11eb-0145-33853c41168d
begin
	x = xs:xs:xmax-xs
	wvnums = [2, 5, 10, 15]
	y = zeros(length(x))
	for w in wvnums
		global y
		y += sin.(2*π*w*x)
	end
	plot(y)
end

# ╔═╡ 62930d90-0773-11eb-090a-df83c32cfc62
function plotSpectrum(x, y)
	F = fft(y) |> fftshift
	freqs = AbstractFFTs.fftfreq(length(x), 1.0/xs) |> fftshift
	freq_domain = plot(freqs, abs.(F), title = "Spectrum", xlim=(0, +50)) 
	plot(freq_domain)
end

# ╔═╡ f2ab3880-0769-11eb-06d6-a947829b1f94
begin
	plotSpectrum(x, y)
	vline!(wvnums)
end

# ╔═╡ b0e96f90-0762-11eb-2619-8de476bf1279
begin
	sub_diag = [-1 for i in 1:(N-1)]
	main_diag = [2 for i in 1:N]
	A = spdiagm(-1 => sub_diag, 0 => main_diag, 1 => sub_diag)
end

# ╔═╡ e9bdb7d0-0768-11eb-2332-a5947e1baa3f
b = zeros(N)

# ╔═╡ 7d86a930-0774-11eb-0b55-bb5b6f8ff278
xinit = rand(N)

# ╔═╡ 3c0a2270-0774-11eb-2f5d-2b3d7793b81d
jacobi!(xinit, A, b)

# ╔═╡ Cell order:
# ╠═0c8ddd80-0764-11eb-3076-f57c55c75d89
# ╠═994548e0-0768-11eb-12cd-21726bbca62f
# ╠═3e1df2d0-076a-11eb-0145-33853c41168d
# ╠═62930d90-0773-11eb-090a-df83c32cfc62
# ╠═f2ab3880-0769-11eb-06d6-a947829b1f94
# ╠═b0e96f90-0762-11eb-2619-8de476bf1279
# ╠═e9bdb7d0-0768-11eb-2332-a5947e1baa3f
# ╠═7d86a930-0774-11eb-0b55-bb5b6f8ff278
# ╠═3c0a2270-0774-11eb-2f5d-2b3d7793b81d
