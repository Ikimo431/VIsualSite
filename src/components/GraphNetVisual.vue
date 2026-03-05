<script setup lang="ts">
import * as protobuf from 'protobufjs'
import { onMounted, watch, ref} from 'vue'
const props = defineProps<{model: string}>()


const neurons = ref<number[]>([])

watch(()=>props.model, async ()=>{
    console.log("IN THE WATCH SELECTED MODEL CHANGED")
    await GetWeights()

})
async function GetWeights(){
    neurons.value = []
    const response = await fetch(`${props.model}.onnx`)
    const buffer = await response.arrayBuffer()
    //console.log("PARSING ONNX OUTPUT")
    //console.log(buffer)

    const root = await protobuf.load('onnx.proto')
    const ModelProto = root.lookupType('onnx.ModelProto')

    const model = ModelProto.decode(new Uint8Array(buffer)) as any
   

    console.log("Initializer count")
    console.log(model.graph.initializer.length)


    let prevLayerName = "input"
    //stores all the weights in raw bytes
    const tensors = model.graph.initializer.filter((t:any) => t.dims.length === 2 && t.name.toLowerCase().includes("weight"))
    .sort((a:any, b:any) => a.name.localeCompare(b.name)) 
    //local compare sorts alphabetically, which works for how layers are named (fc1, fc2,..., output)
    let outFeatures: number = 0
    for (const tensor of tensors){
        if (tensor.dims.length!==2 ||!tensor.name.includes("weight")){
            continue
        }
       
        console.log("NAME: " + tensor.name)
        console.log("DIMS: " + tensor.dims) //shape, ie [16,7] fpr 7 neurons input, 16 neurons hidden layer

        //console.log(tensor.rawData) //raw weights

        const raw = tensor.rawData
        const copy = new Uint8Array(raw.byteLength)
        copy.set(raw)
        let floatWeights = undefined
        try{
             floatWeights = new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength/4)
        }
        catch(e){
            console.log("ERROR making float array: " + e)
        }
        if (!floatWeights){
            break;
        }
        const inFeatures = tensor.dims[1]
        outFeatures = tensor.dims[0]
        if (inFeatures!=null){
            neurons.value = [...neurons.value, inFeatures]
        }
        
        for (let outNeuron = 0; outNeuron<outFeatures; outNeuron++){
            
            for (let inNeuron = 0; inNeuron<inFeatures; inNeuron++){
                const weight = floatWeights[outNeuron*inFeatures + inNeuron]
                console.log(`Weight for ${prevLayerName} neuron ${inNeuron} to ${tensor.name} neuron ${outNeuron} is ${weight}`)
            }
        }
        prevLayerName = tensor.name
    }
    neurons.value=[...neurons.value, outFeatures]
    console.log("Neuron layer sizes: " + JSON.stringify(neurons.value))
}

onMounted(async()=> {
    await GetWeights()
})
    

    
</script>

<template>
    <div class="Layers">
        <div class="Layer" v-for="layer in neurons">
            <div class="Neuron" v-for="i in layer" :key="i">

            </div>
        </div>
    </div>
</template>

<style scoped>
    .Layers{
        display: flex;
        width: 100%;
        align-items: center;
        justify-content: center;
        gap: .5rem;
    }
    .Layer{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 2px solid black;
        gap:.25rem
    }
    .Neuron{
        background-color: gray;
        height: 2rem;
        width: 2rem;
        border-radius: .5rem;
    }

</style>