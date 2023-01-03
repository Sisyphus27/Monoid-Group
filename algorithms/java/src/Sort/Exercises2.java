package Sort;

import static Sort.Sort.mergeSort;

public class Exercises2 {
    public static void main(String[] args) {
        /**
         * 《算法》习题 2.1-5
         *  计算两个长度为n的二进制数组加和，并将结果保存在n+1长度数组中
         */
        System.out.println('\n' + "###########习题 2.1-5###########");
        int[] getArr1 = {1, 1, 0, 1};
        int[] getArr2 = {1, 1, 1, 0};
        int[] res = add(getArr1, getArr2);
        for(int i: res) {
            System.out.print(i); // 测试输出为11011
        }

        /**
         * 《算法》习题 2.3-6
         *  二分查找
         */
        System.out.println('\n' + "###########习题 2.3-6###########");
        int[] arr = {991, 662, 2, 0, -8, 123, 10000, 0};
        int target = 10000;
        int index = binarySearch(arr, target);
        System.out.println(index);

        /**
         * 《算法》习题 2.3-8
         *  给定数 x,判断 Set S 中是否包含两个元素之和为 x ,要求时间复杂度 nlgn
         *  归并排序 + 遍历二分查找
         */
        System.out.println('\n' + "###########习题 2.3-8###########");
        int x = -6;
        boolean flag = findX(arr, x);
        System.out.println(flag);
    }

    private static int[] add(int[] arr1, int[] arr2) {
        int n = arr1.length;
        int[] res = new int[n + 1];
        int temp = 0;
        for(int i = n - 1; i >= 0; i--) {
            int sum = arr1[i] + arr2[i] + temp;
            res[i + 1] = sum % 2;
            temp = sum / 2;
        }
        res[0] = temp;
        return res;
    }

    public static int binarySearch(int[] arr, int target) {
        int left = 0;
        int right = arr.length - 1;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(arr[mid] == target)
                return mid;
            else if (arr[mid] < target)
                left = mid + 1;
            else if (arr[mid] > target)
                right = mid - 1;
        }
        return -1;
    }

    public static boolean findX(int[] arr, int x) {
        int n = arr.length;
        int[] mergeSortedArr = mergeSort(arr, 0, (n - 1) / 2, n - 1);
        for(int i = 0; i < n; i++) {
            int key = mergeSortedArr[i];
            int index = binarySearch(mergeSortedArr, x - key);
            if(index != -1 && index != i){
                return true;
            }
        }
        return false;
    }
}
