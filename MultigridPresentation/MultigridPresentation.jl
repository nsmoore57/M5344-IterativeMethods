### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 4e6859e8-07ec-11eb-3200-5d7441db3541
begin
	import Pkg
	Pkg.activate(".")
	using IterativeSolvers
	using SparseArrays
	using Plots
	using LinearAlgebra
end

# ╔═╡ f8619160-07ec-11eb-28eb-e12b3f552197
md"# Multigrid"

# ╔═╡ b2f84422-08da-11eb-15b0-ef0e45dab69c
md"## Analyzing the Jacobi Method"

# ╔═╡ d4dc33ee-07ec-11eb-0bfd-039484b81720
md"First, create the A matrix, this will correspond to a 1D poisson problem."

# ╔═╡ 268fd020-08e3-11eb-04b5-37f4a48a1e6e
N = 1000

# ╔═╡ bedc360e-08da-11eb-29e8-45bc78190706
A = spdiagm(-1=>-ones(N-1), 0=>2*ones(N), 1=>-ones(N-1))

# ╔═╡ 03b38c4e-07ed-11eb-2200-1b381ca5c349
md"For the sake of demonstration, we let the right-hand side vector b correspond to the case of Dirichlet boundary conditions where both ends are held at 0. We choose this because the true solution is then the zero vector and so the error cooresponds with the current iterate."

# ╔═╡ 640fb614-07eb-11eb-25e9-c1501af3e6bb
b = zeros(N)

# ╔═╡ 38a285ec-08db-11eb-3dbf-e52720123bb3
md"Let's generate some initial guesses (which are also initial errors) which are sine waves with varying frequencies"

# ╔═╡ d2e7080e-08da-11eb-1280-4de82702e49d
begin
	x = range(0,stop=1, length=N)
	waveNumbers = [1, 3, 7, 20, 100]
	xinitial = [sin.(w*π*x) for w in waveNumbers]
end

# ╔═╡ 2846866c-08db-11eb-3287-5784dac4644b
plot(x, xinitial, layout=(length(xinitial),1), legend=false)

# ╔═╡ 2b55bab6-08dc-11eb-3f22-21afeaa4ea44
numJacobiIters = 10000

# ╔═╡ b9c1a3c4-08db-11eb-2e91-cb4241130b44
md"Now let's run $numJacobiIters Jacobi iterations on each of these, tracking the error at each iteration."

# ╔═╡ 2128b458-08dc-11eb-046e-97261b58dd8b
begin
	errors = [zeros(numJacobiIters) for i ∈ waveNumbers]
	results = deepcopy(xinitial)
	for j ∈ 1:numJacobiIters
		for (i, w) ∈ enumerate(results)
			errors[i][j] = norm(jacobi!(w, A, zeros(size(w)), maxiter=1))
		end
	end
end

# ╔═╡ 8e751486-08e4-11eb-377b-7b9169d0d51c
md"Here we plot these errors"

# ╔═╡ fac138a2-08de-11eb-1f75-39183f2801ec
begin
	labels = reshape([string(w) for w in waveNumbers], 1, :)
	plot(errors, label=labels, xlabel="Iteration", ylabel="error")
end

# ╔═╡ f1915512-08e3-11eb-1e3f-cf0a3a44e916
md"We can also plot the actual results of the iterations"

# ╔═╡ 016ca266-08e4-11eb-18c6-09736686a7e2
plot(x, results, layout=(length(xinitial),1), legend=false)

# ╔═╡ 96db8162-08e3-11eb-36ad-6d583ec2234f
md"Here we see the key to understanding the effectiveness of multigrid. The Jacobi iteration scheme is much better at eliminating high frequency error than low frequency error."

# ╔═╡ c5ec024a-07ec-11eb-3868-49c7229e0e7d
md"## Appendix"

# ╔═╡ 812b3720-08e3-11eb-012a-9b0beb675bfb
md"Imports"

# ╔═╡ Cell order:
# ╟─f8619160-07ec-11eb-28eb-e12b3f552197
# ╟─b2f84422-08da-11eb-15b0-ef0e45dab69c
# ╟─d4dc33ee-07ec-11eb-0bfd-039484b81720
# ╠═268fd020-08e3-11eb-04b5-37f4a48a1e6e
# ╟─bedc360e-08da-11eb-29e8-45bc78190706
# ╟─03b38c4e-07ed-11eb-2200-1b381ca5c349
# ╟─640fb614-07eb-11eb-25e9-c1501af3e6bb
# ╟─38a285ec-08db-11eb-3dbf-e52720123bb3
# ╠═d2e7080e-08da-11eb-1280-4de82702e49d
# ╟─2846866c-08db-11eb-3287-5784dac4644b
# ╟─b9c1a3c4-08db-11eb-2e91-cb4241130b44
# ╠═2b55bab6-08dc-11eb-3f22-21afeaa4ea44
# ╠═2128b458-08dc-11eb-046e-97261b58dd8b
# ╟─8e751486-08e4-11eb-377b-7b9169d0d51c
# ╟─fac138a2-08de-11eb-1f75-39183f2801ec
# ╟─f1915512-08e3-11eb-1e3f-cf0a3a44e916
# ╟─016ca266-08e4-11eb-18c6-09736686a7e2
# ╟─96db8162-08e3-11eb-36ad-6d583ec2234f
# ╟─c5ec024a-07ec-11eb-3868-49c7229e0e7d
# ╟─812b3720-08e3-11eb-012a-9b0beb675bfb
# ╠═4e6859e8-07ec-11eb-3200-5d7441db3541
