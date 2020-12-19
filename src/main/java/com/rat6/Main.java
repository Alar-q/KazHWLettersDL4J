package com.rat6;


import org.apache.log4j.BasicConfigurator;
import org.bytedeco.opencv.opencv_java;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;

import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.io.*;
import java.util.List;
import java.util.Random;



public class Main {
    public static void main(String[] args) throws IOException{
        BasicConfigurator.configure();
        //new opencv_java();
        new Nd4j();

        int WIDTH = 40;
        int labelsSum = 42;
        int batchSize = 64;

        String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        Random rand = new Random();
        File root = new File("C:\\Users\\User\\Documents\\kazLetRedacted");
        FileSplit f = new FileSplit(root, allowedExtensions, rand);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader reader;
        DataSetIterator data_train;
        reader = new ImageRecordReader(WIDTH, WIDTH, 1, labelMaker);
        reader.initialize(f);

        data_train = new RecordReaderDataSetIterator(reader, batchSize, 1, labelsSum);

        /**/
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(321)
                .activation(Activation.SIGMOID)
                .miniBatch(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.05))
                .biasInit(0)
                .weightInit(new org.deeplearning4j.nn.conf.distribution.UniformDistribution(-1, 1))

                .list()
                .layer(new DenseLayer.Builder().nIn(WIDTH*WIDTH).nOut(WIDTH*WIDTH/2)
                        .build())
                .layer(new DenseLayer.Builder().nOut(WIDTH*WIDTH/2)
                        .build())
                .layer(new DenseLayer.Builder().nOut(WIDTH*WIDTH/4)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                         .activation(Activation.SOFTMAX)
                         .nOut(labelsSum).build())
                .setInputType(InputType.convolutional(WIDTH,WIDTH,1))
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();


        network.setListeners(new ScoreIterationListener(10));
        System.out.println(network.summary());

        // here the actual learning takes place
        for( int i=0; i < 500; i++ )
            network.fit(data_train);


        root = new File("C:\\Users\\User\\Documents\\kazLetRedacted");
        f = new FileSplit(root, allowedExtensions, rand);
        labelMaker = new ParentPathLabelGenerator();
        reader = new ImageRecordReader(WIDTH, WIDTH, 1, labelMaker);
        reader.initialize(f);

        DataSetIterator data_test = new RecordReaderDataSetIterator(reader, batchSize, 1, labelsSum);;

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(labelsSum);
        while(data_test.hasNext()){
            DataSet t = data_test.next();
            INDArray features = t.getFeatures();
            INDArray lables = t.getLabels();
            INDArray predicted = network.output(features,false);
            eval.eval(lables, predicted);
        }
        System.out.println(eval.stats());


        network.save(new File("C:\\weights\\kazletters_model.zip"));

    }
}
