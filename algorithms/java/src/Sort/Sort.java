package Sort;

public class Sort {
    public static void main(String[] args) {
        int[] arr = {991, 662, 2, 0, -8, 123, 10000, 0};
        int n = arr.length;
        System.out.println("插入排序（非递归版）：");
        int[] insertSortedArr = insertSort(arr, n);
        for(int i: insertSortedArr) {
            System.out.print(i + " ");
        }
        System.out.println('\n' + "插入排序（递归版）：");
        int[] insertSortedRecursiveArr = insertSortRecursive(arr, n);
        for(int i: insertSortedRecursiveArr) {
            System.out.print(i + " ");
        }
        System.out.println('\n' + "选择排序：");
        int[] selectSortedArr = selectSort(arr);
        for(int i: selectSortedArr) {
            System.out.print(i + " ");
        }
        System.out.println('\n' + "归并排序(非递归版)：");
        int[] mergeSortedArr = mergeSort(arr, 0, (n - 1) / 2, n - 1);
        for(int i: mergeSortedArr) {
            System.out.print(i + " ");
        }
        System.out.println('\n' +"归并排序(递归版)：");
        int[] mergeSortedRecursiveArr = mergeSortRecursive(arr, 0, n - 1);
        for(int i: mergeSortedRecursiveArr) {
            System.out.print(i + " ");
        }
    }

    /**
     * 插入排序 非递归版
     * @param arr
     * @return
     */
    private static int[] insertSort(int[] arr, int n) {
        if(n <= 1) return arr;
        for(int i = 1; i < n; i++){
            int key = arr[i];
            int j = i - 1;
            while(j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
        return arr;
    }

    /**
     * 插入排序 递归版
     * @param arr
     * @return
     */
    private static int[] insertSortRecursive(int[] arr, int n) {
        //递归出口
        if(n <= 1) return arr;
        //对前 n - 1 个元素排序
        insertSortRecursive(arr,n - 2);
        //把位置 n 的元素插入到前面的部分
        int key = arr[n - 1];
        int j = n - 2;
        while(j >= 0 && arr[j] > key) {//小的往前插
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
        return arr;
    }

    /**
     * 选择排序
     * @param arr
     * @return
     */
    private static int[] selectSort(int[] arr) {
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

    /**
     * 归并排序 非递归版
     * @param arr
     * @return
     */
    static int[] mergeSort(int[] arr, int p, int q, int r) {
        int ln = q - p + 1;
        int rn = r - q;
        int[] L = new int[ln];
        int[] R = new int[rn];
        for(int i = 0; i < ln; i++) {
            L[i] = arr[p + i];
        }
        for(int j = 0; j < rn; j++) {
            R[j] = arr[q + j + 1];
        }
        int i = 0, j = 0, k = p;
        while(i < ln && j < rn) {
            if(L[i] <= R[j]) {
                arr[k] = L[i];
                i++;
            }
            else {
                arr[k] = R[j];
                j++;
            }
            k++;
        }
        while(i < ln) {
            arr[k] = L[i];
            i++;
            k++;
        }
        while(j < rn) {
            arr[k] = R[j];
            j++;
            k++;
        }
        return arr;
    }

    /**
     * 归并排序 递归版
     * @param arr
     * @return
     */
    private static int[] mergeSortRecursive(int[] arr, int p, int r) {
        if(p >= r) return arr;
        int q = (p + r) /2;
        mergeSortRecursive(arr, p, q);
        mergeSortRecursive(arr, q + 1, r);
        return mergeSort(arr, p, q, r);
    }

}
