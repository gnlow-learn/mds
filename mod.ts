import * as np from "https://esm.sh/numpy-ts@1.1.0"

const mds = (D: np.NDArray, k = 2) => {
    const n = D.shape[0]
    const J = np.eye(n).subtract(np.ones([n, n]).divide(n))
    const B = J.matmul(D.power(2)).matmul(J).multiply(-0.5)

    const { w, v } = np.linalg.eigh(B)
    const idx = np.flip(np.argsort(w))
    
    const topIdx = np.take(idx, np.arange(0, k).toArray(), 0).toArray()
    const topEigVals = np.maximum(np.take(w, topIdx, 0), 0)
    const topEigVecs = np.take(v, topIdx, 1)

    return topEigVecs.multiply(topEigVals.sqrt())
}

const distMatrix = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
])

const coords = mds(distMatrix, 2)
console.log(coords.toArray())
