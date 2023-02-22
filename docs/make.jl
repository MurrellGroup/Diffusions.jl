using Diffusions
using Documenter

DocMeta.setdocmeta!(Diffusions, :DocTestSetup, :(using Diffusions); recursive=true)

makedocs(;
    modules=[Diffusions],
    authors="Ben Murrell <murrellb@gmail.com> and contributors",
    repo="https://github.com/murrellb/Diffusions.jl/blob/{commit}{path}#{line}",
    sitename="Diffusions.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://murrellb.github.io/Diffusions.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/murrellb/Diffusions.jl",
    devbranch="main",
)
