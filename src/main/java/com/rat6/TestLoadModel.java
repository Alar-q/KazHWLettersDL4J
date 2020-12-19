package com.rat6;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class TestLoadModel {
    public static void main(String[] args) throws IOException {
        BasicConfigurator.configure();

        new Nd4j();

        MultiLayerNetwork network = MultiLayerNetwork.load(new File("C:\\weights\\bu.zip"), true);

        String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        Random rand = new Random();
        File root = new File("C:\\pathTo2");
        FileSplit f = new FileSplit(root, allowedExtensions, rand);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader reader;
        DataSetIterator data;
        reader = new ImageRecordReader(32, 32, 1, labelMaker);
        reader.initialize(f);

        data = new RecordReaderDataSetIterator(reader, 64, 1, 10);

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(10);
        while(data.hasNext()){
            DataSet t = data.next();
            INDArray features = t.getFeatures();
            INDArray lables = t.getLabels();
            INDArray predicted = network.output(features,false);
            eval.eval(lables, predicted);
        }
        System.out.println(eval.stats());

    }
}
