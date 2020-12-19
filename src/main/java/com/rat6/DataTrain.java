package com.rat6;

import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.io.File;
import java.util.Random;

import static org.deeplearning4j.nn.conf.Updater.NESTEROVS;

public class DataTrain {
    private static final Logger log = LoggerFactory.getLogger(DataTrain.class);
    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();
        new Nd4j();
        double learningRate = 0.05; // Число входных каналов
        int nChannels = 1; // Число входных каналов

        int outputNum = 42; // Число возможных исходов
        int batchSize = 64; // Размер тестового пакета
        int nEpochs = 10; // Число периодов обучения
        int seed = 123;
        int WIDTH = 32;

        int[] inputShape = new int[] {nChannels, WIDTH, WIDTH};

        log.info("Загружаются данные....");
        String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        Random rand = new Random();
        File root = new File("C:\\Users\\Pupochek\\Documents\\kazLetRedacted");
        FileSplit f = new FileSplit(root, allowedExtensions, rand);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader reader;
        DataSetIterator data;
        reader = new ImageRecordReader(WIDTH, WIDTH, 1, labelMaker);
        reader.initialize(f);

        data = new RecordReaderDataSetIterator(reader, batchSize, 1, outputNum);


        log.info("Построение модели....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.1, 0.9))

                .list()
                // block 1
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn1")
                        .nIn(inputShape[0])
                        .stride(1, 1)
                        .nOut(6)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("maxpool1")
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // block 2
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .name("cnn2")
                        .stride(1, 1)
                        .nOut(16)
                        .activation(Activation.RELU).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .name("maxpool2")
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // fully connected
                .layer(4, new DenseLayer.Builder()
                        .name("ffn1")
                        .activation(Activation.RELU)
                        .nOut(120)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .name("ffn1")
                        .activation(Activation.RELU)
                        .nOut(84)
                        .build())

                // output
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX) // radial basis function required
                        .build())
                .setInputType(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Обучение модели....");
        model.setListeners(new ScoreIterationListener(10));

        for (int i = 0; i < nEpochs; i++) {
            System.out.println("epoch: " + i);
            model.fit(data);
        }



        allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        root = new File("C:\\Users\\Pupochek\\Documents\\kazLetRedacted");
        f = new FileSplit(root, allowedExtensions, rand);
        labelMaker = new ParentPathLabelGenerator();

        reader = new ImageRecordReader(WIDTH, WIDTH, 1, labelMaker);
        reader.initialize(f);
        data = new RecordReaderDataSetIterator(reader, batchSize, 1, outputNum);

        Evaluation eval = new Evaluation(outputNum);
        while (data.hasNext()) {
            DataSet ds = data.next();
            INDArray output = model.output(ds.getFeatures(), false);
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
        data.reset();

        model.save(new File("C:\\weights\\kazlettersLeNet.zip"));

        log.info("****************Конец обучения********************");
    }
}
