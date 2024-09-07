func.func @nested_loop_reduction(%N : index, %M : index, %L : index, 
                                 %A : memref<?x?x?xf32>, 
                                 %B : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %N step %c1 {
    scf.for %j = %c0 to %M step %c1 {
      scf.for %k = %c0 to %L step %c1 {
        %val = memref.load %A[%i, %j, %k] : memref<?x?x?xf32>
        %prev = memref.load %B[%i, %j] : memref<?x?xf32>
        %sum = arith.addf %prev, %val : f32
        memref.store %sum, %B[%i, %j] : memref<?x?xf32>
      }
    }
  }
  return
}