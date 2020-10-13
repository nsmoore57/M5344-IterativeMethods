### A Pluto.jl notebook ###
# v0.12.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 4e6859e8-07ec-11eb-3200-5d7441db3541
begin
	import Pkg
	Pkg.activate(".")
	using IterativeSolvers
	using SparseArrays
	using Plots
	using LinearAlgebra
	using PlutoUI
end

# ╔═╡ f8619160-07ec-11eb-28eb-e12b3f552197
md"# Multigrid"

# ╔═╡ b2f84422-08da-11eb-15b0-ef0e45dab69c
md"## Analyzing the Jacobi Method"

# ╔═╡ d4dc33ee-07ec-11eb-0bfd-039484b81720
md"First, create the A matrix, this will correspond to a 1D poisson problem."

# ╔═╡ 268fd020-08e3-11eb-04b5-37f4a48a1e6e
N = 1024

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

# ╔═╡ 8ad57950-0903-11eb-16a4-c30fbf400ff4
md"## How do we use this?"

# ╔═╡ 77644ac0-0905-11eb-012a-17d36449b91e
md"The Jacobi property of efficiently removing high frequency error works independent of the length of the vector.  Notice what happens if we start with a low frequency wave and approximate it by half as many points."

# ╔═╡ d8c14290-0906-11eb-1b87-bde93f22a2ad
@bind waveNum Slider(1:512, show_value=true)

# ╔═╡ 19fdf6a0-0906-11eb-28ae-7b8f11724e66
begin 
	y_fine = sin.(waveNum*π*x)
	x_coarse = [x[i] for i ∈ 1:length(x) if i % 2 == 0]
	y_coarse = sin.(waveNum*π*x_coarse)
	plot([y_fine, y_coarse], layout=(2,1), xlim=(0,N), legend=false)
end

# ╔═╡ d1052080-0906-11eb-14ea-df7dac9a2fd4
md"Notice that the same number of waves now fit into half the length.  This means that Jacobi iteration could be more effective on this new, shorter vector."

# ╔═╡ 6ee2b5d0-0905-11eb-1a06-0f0c81dfe53d
md"## In Practice"

# ╔═╡ e6efcbd0-090a-11eb-2525-ad2c03d72d22
md"Consider a discretized PDE problem on a grid (which we'll denote $\Omega^h$) where $h$ represents the spacing between nodes. As the name suggests, for the multigrid method we'll be using multiple grids, each with a different spacing of nodes. From here on, we'll be using superscript to denote which grid a quantity is on."

# ╔═╡ d5f35ce0-0903-11eb-1ade-a3c30cf8fd71
md"Our discretized problem is written as $A^h x^h = b^h$.  We'll start with $k$ Jacobi iterations. Since we don't expect our current iteration $x^h_k$ to be the exact solution, let's assume the exact solution is of the form $x^h_k + \varepsilon^h$.  This gives us an equation of the form"

# ╔═╡ a2de02f0-0904-11eb-04b8-c5d9412de529
md"$A^h(x^h_k + \varepsilon^h) = b^h$"

# ╔═╡ b9631650-0904-11eb-06c1-5b103388af03
md"Rearranging this equation gives"

# ╔═╡ ca7a1650-0904-11eb-25bf-87b9b4f006d5
md"$b^h - A^hx^h_k = A^h\varepsilon^h = r^h_k$"

# ╔═╡ 36973de0-0905-11eb-280f-3f2078e12ad5
md"So if we calculate $r^h_k$ and solve $A^h\varepsilon^h = r^h_k$ for $\varepsilon^h$, then we could find the exact solution as $x^h_k + \varepsilon^h$."

# ╔═╡ 953c92f0-090a-11eb-023f-6b21d8929ed9
md"So how do we find or (more accurately) approximate $\varepsilon^h$?  Running more Jacobi iterations at this level has already shown to be less effective since the high frequency error has already been removed. Only the lower frequency error remains. Instead, we will move the problem down to a coarser grid, $\Omega^{2h}$. In the coarser grid, the low frequency error changes to higher frequency error and Jacobi can be more effective."

# ╔═╡ a2a625e0-090b-11eb-03b9-49fd3d6c7caf
md"That is, we want to solve $A^{2h}e^{2h} = r^{2h}_k$, where $A^{2h}$, $e^{2h}$, and $r^{2h}_k$ are the \"coarse grid versions\" of $A^h$, $e^h$, and $r^h_k$. We will discuss how to find these later."

# ╔═╡ 83a358b0-090c-11eb-2299-07cd73dcc28c
md"We can now use Jacobi on this equation and the high frequency error in $\Omega^{2h}$ (which is the lower frequency error in $\Omega^{h}$) will be removed."

# ╔═╡ 1f15cf80-090d-11eb-0884-d5c43fb5943d
md"After running iterations of Jacobi on $\Omega^{2h}$ to get an approximation for $e^{2h}$, we then \"transfer\" this back into the $\Omega^h$ grid and it becomes an approximation to $e^h$.  We then calculate $x^h_k + e^h$ to get a better approximation for $x^h_k$. In doing so, the transfer may have introduced more high frequency error, so we typically complete more Jacobi iterations at the fine level to remove these. This process leverages the change of grids to use Jacobi iteration more effectively."

# ╔═╡ b31d7890-090d-11eb-1928-37873fb68fbe
md"## A Formal Two-Grid Cycle"

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
# ╟─2b55bab6-08dc-11eb-3f22-21afeaa4ea44
# ╠═2128b458-08dc-11eb-046e-97261b58dd8b
# ╟─8e751486-08e4-11eb-377b-7b9169d0d51c
# ╟─fac138a2-08de-11eb-1f75-39183f2801ec
# ╟─f1915512-08e3-11eb-1e3f-cf0a3a44e916
# ╟─016ca266-08e4-11eb-18c6-09736686a7e2
# ╟─96db8162-08e3-11eb-36ad-6d583ec2234f
# ╟─8ad57950-0903-11eb-16a4-c30fbf400ff4
# ╟─77644ac0-0905-11eb-012a-17d36449b91e
# ╟─d8c14290-0906-11eb-1b87-bde93f22a2ad
# ╠═19fdf6a0-0906-11eb-28ae-7b8f11724e66
# ╟─d1052080-0906-11eb-14ea-df7dac9a2fd4
# ╟─6ee2b5d0-0905-11eb-1a06-0f0c81dfe53d
# ╟─e6efcbd0-090a-11eb-2525-ad2c03d72d22
# ╟─d5f35ce0-0903-11eb-1ade-a3c30cf8fd71
# ╟─a2de02f0-0904-11eb-04b8-c5d9412de529
# ╟─b9631650-0904-11eb-06c1-5b103388af03
# ╟─ca7a1650-0904-11eb-25bf-87b9b4f006d5
# ╟─36973de0-0905-11eb-280f-3f2078e12ad5
# ╟─953c92f0-090a-11eb-023f-6b21d8929ed9
# ╟─a2a625e0-090b-11eb-03b9-49fd3d6c7caf
# ╟─83a358b0-090c-11eb-2299-07cd73dcc28c
# ╟─1f15cf80-090d-11eb-0884-d5c43fb5943d
# ╟─b31d7890-090d-11eb-1928-37873fb68fbe
# ╟─c5ec024a-07ec-11eb-3868-49c7229e0e7d
# ╟─812b3720-08e3-11eb-012a-9b0beb675bfb
# ╠═4e6859e8-07ec-11eb-3200-5d7441db3541
