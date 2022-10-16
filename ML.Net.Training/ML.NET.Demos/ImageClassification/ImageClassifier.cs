using ImageClassification.MachineLearning;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;

namespace ImageClassification;

public class ImageClassifier
{
    private static string _modelPath => Path.Combine(AppContext.BaseDirectory, "imageClassification.mdl");

    private readonly MLContext _mlContext;
    private readonly ImageClassificationTrainer.Architecture _architecture;
    private readonly DataLoader _dataLoader;
    private ITransformer _trainedModel;

    public ImageClassifier(ImageClassificationTrainer.Architecture architecture, DataLoader dataLoader)
    {
        _mlContext = new(111);
        _architecture = architecture;
        _dataLoader = dataLoader;
    }

    public void Fit()
    {
        var trainingPipeline = BuildTrainingPipeline();
        _trainedModel = trainingPipeline.Fit(_dataLoader.TrainSet);
    }

    public void Save() =>
        _mlContext.Model.Save(_trainedModel, _dataLoader.TrainSet.Schema, _modelPath);

    private EstimatorChain<KeyToValueMappingTransformer> BuildTrainingPipeline()
    {
        var classifierOptions = new ImageClassificationTrainer.Options()
        {
            FeatureColumnName = "Image",
            LabelColumnName = "LabelAsKey",
            ValidationSet = _dataLoader.ValidationSet,
            Arch = _architecture,
            MetricsCallback = (metrics) => Console.WriteLine(metrics),
            TestOnTrainSet = false,
            ReuseTrainSetBottleneckCachedValues = true,
            ReuseValidationSetBottleneckCachedValues = true,
            Epoch = 20
        };

        return _mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
    }
}
