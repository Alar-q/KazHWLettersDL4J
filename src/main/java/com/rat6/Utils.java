package com.rat6;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;

public class Utils {

    public static void listnData(DataSetIterator data, Mat mat, int i){
        INDArray indArray = data.next(1).getFeatures();
        //INDArray indArray = data.next().getFeatures();
        ndarr2Mat(indArray, mat);
        Core.multiply(mat, new Scalar(1d/255d), mat);
        System.out.println(i+" "+mat.dump());
    }

    public static Mat ndarr2Mat(INDArray array, Mat mat){
        int a = (int) array.length();
        double[] arr = new double[a];
        for (int i = 0; i < arr.length; i++)
            arr[i] = array.getDouble(i);
        mat.put(0, 0, arr);
        return mat;
    }


    //uncorrect
    public static void mattond4j() throws IOException {
        Mat orig = Imgcodecs.imread("C:\\pathTo2\\0\\00.png");
        Mat I = new Mat();
        Imgproc.cvtColor(orig, I, COLOR_BGR2GRAY);
        System.out.println(I.dump());

        NativeImageLoader imageLoader = new NativeImageLoader(32, 32, 1);
        INDArray array = imageLoader.asMatrix(I);
        System.out.println(array);

        int leng = (int) array.length();
        double[] arr = new double[leng];
        for(int i=0; i<leng; i++)
            arr[i] = array.getDouble(i);

        Mat mat = new Mat(32, 32, CvType.CV_64FC1);
        mat.put(0, 0, arr);
        System.out.println(mat.dump());
    }
    public static void readdataooo(){
        //Прочитать data
        DataSetIterator data = null;
        for(int j=1; j<2; j++) {
            INDArray indArray = data.next(j).getFeatures();
            int a = (int) indArray.length();
            double[] arr = new double[a];

            for (int i = 0; i < arr.length; i++)
                arr[i] = indArray.getDouble(i);

            Mat in = new Mat(32, 32, CvType.CV_64FC1);
            in.put(0, 0, arr);

            Mat in1 = new Mat();
            Core.multiply(in, new Scalar(1d/255d), in1);

            System.out.println(in1.dump());
            in.release();
            in1.release();
            indArray.close();
        }
    }
}
