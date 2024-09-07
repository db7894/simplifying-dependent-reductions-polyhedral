func.func @prefix_sum(%N : index, %A: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  scf.for %i = %c0 to %N step %c1 {
    %sum_init = arith.constant 0.0 : f32
    %sum = scf.for %j = %c0 to %i step %c1 iter_args(%sum_iter = %sum_init) -> (f32) {
      %a_j = memref.load %A[%j] : memref<?xf32>
      %sum_next = arith.addf %sum_iter, %a_j : f32
      scf.yield %sum_next : f32
    }
    memref.store %sum, %B[%i] : memref<?xf32>
  }
  
  scf.for %i = %c0 to %N step %c1 {
    %b_i = memref.load %B[%i] : memref<?xf32>
    %i_plus_1 = arith.addi %i, %c1 : index
    %a_next = func.call @f(%b_i) : (f32) -> f32
    memref.store %a_next, %A[%i_plus_1] : memref<?xf32>
  }
  
  return
}

func.func private @f(%arg0: f32) -> f32