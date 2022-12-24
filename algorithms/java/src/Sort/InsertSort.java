package Sort;

public class InsertSort {
    public static void main(String[] args) {
        int[] arr = {991, 662, 2, 0, -8, 123, 10000};
        int[] sorted_arr = sort(arr);
        for(int i: sorted_arr) {
            System.out.println(i);
        }
    }

    // 升序排列
    private static int[] sort(int[] arr) {
        int n = arr.length;
        if(n <= 1) return arr;
        for(int i = 1; i < n; i++){
            int key = arr[i];
            int j = i - 1;
            while(j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j -= 1;
            }
            arr[j + 1] = key;
        }
        return arr;
    }

}
