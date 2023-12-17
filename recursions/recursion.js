function rec(x){
console.log(x);
if(x==1){
return 1;
}else{
return x*rec(x-1);
}
}

result=rec(5);

console.log(result);