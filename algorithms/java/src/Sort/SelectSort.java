package Sort;

public class SelectSort {
    public static void main(String[] args) {
        int[] arr = {991, 662, 2, 0, -8, 123, 10000};
        int[] sorted_arr = sort(arr);
        for (int i : sorted_arr) {
            System.out.println(i);
        }
    }
    private static int[] sort(int[] arr) {
        int n = arr.length;
        if(n <= 1) return arr;
        for(int i = 0; i < n; i++) {
            int min = i;
            for(int j = i + 1; j < n; j++){
                if(arr[j] < arr[min]) {
                    min = j;
                }
            }
            int temp = arr[min];
            arr[min] = arr[i];
            arr[i] = temp;
            //位运算实现交换 在0处会出现错误
//            arr[min] = arr[min] ^ arr[i];
//            arr[i] =  arr[min] ^ arr[i];
//            arr[min] = arr[min] ^ arr[i];
        }
        return arr;
    }
}
