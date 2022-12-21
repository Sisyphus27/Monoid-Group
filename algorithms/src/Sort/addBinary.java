package Sort;

/**
 * 《算法》习题 2.1-5
 *  计算两个长度为n的二进制数组加和，并将结果保存在n+1长度数组中
 */
public class addBinary {
    public static void main(String[] args) {
        int[] getArr1 = {1, 1, 0, 1};
        int[] getArr2 = {1, 1, 1, 0};
        int[] res = add(getArr1, getArr2);
        for(int i: res) {
            System.out.print(i); // 测试输出为11011
        }
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
}
