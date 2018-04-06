

for p in ("Knet","ArgParse", "Compat", "GZip", "Images","ImageCore")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet, Compat,GZip, ArgParse, Images, ImageCore

include(Pkg.dir("Knet","data","mnist.jl"))


function main(args=ARGS)
    s = ArgParseSettings()
    s.description="Generative Adversarial Networks Knet Implementation."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=32; help="minibatch size")
        ("--lr"; arg_type=Float64; default=0.2; help="learning rate")
        ("--k"; arg_type=Int; default=1; help="k")
        ("--geninputsize"; arg_type=Int; default=784; help="size of the generator's input")
        ("--genhidden"; arg_type=Int; default=512; help="sizes of the generator hidden layer")
        ("--dischidden"; arg_type=Int; default=256; help="sizes of the discriminator hidden layer")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--epochs"; arg_type=Int; default=3; help="number of epochs for training")
        ("--iters"; arg_type=Int; default=typemax(Int); help="maximum number of updates for training")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array and float type to use")
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    println(s.description)
    o = parse_args(args, s; as_symbols=true)
    println("o=",[(k,v) for (k,v) in o]...)
    o[:seed] > 0 && srand(o[:seed])
    global atype = eval(parse(o[:atype]))
    # if atype <: Array; warn("CPU conv4 support is experimental and very slow."); end

    xtrn,ytrn,xtst,ytst = mnist()
    global dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=atype)
    global dtst = minibatch(xtst, ytst, o[:batchsize]; xtype=atype)
    imsize = 28*28

    w = initweights(o[:genhidden], o[:geninputsize], imsize)
    discw = initweights(o[:dischidden], imsize, 1)
    append!(w, discw)

    println(avgloss(w,dtst,o))
    train!(w, dtrn, o)
    println(avgloss(w,dtst,o))
    generate(w,10,o)

    return w,o,dtrn,dtst
end


function generate(w,ninstance,o;genfolder="gen/")
    z = samplez(o[:geninputsize],ninstance)
    gz = generator(w[1:4],z)
    gz = reshape(gz,(28,28,ninstance))
    gz = gz .> 0.5
    gz = permutedims(gz,(2,1,3))
    # imtype = Array{Gray{N0f8},2}
    for i=1:ninstance
        # save(genfolder*string(i)*".png",convert(imtype,gz[:,:,i]))
        # A = rand(3, 10, 10)
        img = colorview(Gray, convert(Array{Float32,2},gz[:,:,i]))
        save(genfolder*string(i)*".png", img)
    end
end

function avgloss(model,data,o)
    b = o[:batchsize]
    geninputsize = o[:geninputsize]
    total=dloss=gloss=0.0
    for (x,_) in data
        for k = 1:o[:k]
            z = samplez(geninputsize,b)
            gz = generator(model[1:4], z)
            dloss += 2*b*discloss(model[5:end],x,gz)
        end
        z = samplez(geninputsize,2*b)
        gloss += 2*b*genloss(model[1:4], model[5:end], z)
        total += 2*b
    end

    return dloss/total,gloss/total
end

function train!(model, data, o)
    b = o[:batchsize]
    geninputsize = o[:geninputsize]
    for i=1:o[:epochs]
        for (x,_) in data
            for k = 1:o[:k]
                z = samplez(geninputsize,b)
                gz = generator(model[1:4], z)
                g = discgradient(model[5:end],x,gz)
                update!(model[5:end], g, lr = o[:lr])
            end
            z = samplez(geninputsize,2*b)
            g = gengradient(model[1:4], model[5:end], z)
            update!(model[1:4], g, lr = o[:lr])
        end
        println(avgloss(model,dtrn,o))
    end
    # @save "genparams.jld2" model
    # save("ganparam.jld",model)
end


function discloss(w,x,gz)
    dx = discriminator(w,x)
    dz = discriminator(w,gz)
    loss = log.(1e-9 .+ dx) + log.(1e-9 + 1 .- dz)
    return -mean(loss)/2.0
end
discgradient = grad(discloss)

function genloss(wgen, wdisc, z)
    gz = generator(wgen,z)
    dz = discriminator(wdisc,gz)
    loss = -log.(1e-9 .+ dz)
    return mean(loss)
end
gengradient = grad(genloss)

function discriminator(w,x)
    for i=1:2:length(w)
       x = w[i]*mat(x) .+ w[i+1]
       if i<length(w)-1
           x = leakyrelu(x) # max(0,x)
       end
   end
   return sigm.(x)
end

function generator(w,z)
    for i=1:2:length(w)
       z = w[i]*mat(z) .+ w[i+1]
       if i<length(w)-1
           z = leakyrelu(z) # max(0,x)
       end
   end
   return sigm.(z)
end

function leakyrelu(x;alpha = 0.2)
    return max.(0,x) + min.(0,x) * alpha
end


function initweights(hidden, ninputs, noutputs; winit=1)
    w = Any[]
    x = ninputs
    for y in [hidden...,noutputs]
        push!(w, convert(atype, winit*xavier(y,x)))
        push!(w, convert(atype, zeros(y, 1)))
        x = y
    end
    return w
end


function samplez(geninputsize, batch)
    convert(atype, randn(Float32, geninputsize, batch))
end

main(ARGS)
