import { np } from "../deps.ts"
import { mds, getStress } from "../mod.ts"

const distMatrix = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
])

const coords = mds(distMatrix, 2)
console.log(coords.toArray())
console.log(getStress(distMatrix)(coords))
