function recSum(arr){
    if(arr.length === 1){
        return arr[0];
    }
    else{
        return arr.pop() + recSum(arr);
    }

}

arr = [1,2,3,4,5,6,7,8,9,10];
recSum(arr);