function recSum(arr){
    if(arr.length === 1){
        return 1;
    }
    else{
     arr.pop();
     return   1 + recSum(arr);
    }

}

arr = [1,2,3,4,5,6,7,8,9,10];
let result = recSum(arr);

console.log(result);