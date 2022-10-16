using ImageClassification.MachineLearning;
using ImageClassification.MachineLearning.DataModels;
using Microsoft.ML;

namespace ImageClassification;

public class Predictor
{
    private static string _modelPath => Path.Combine(AppContext.BaseDirectory, "imageClassification.mdl");
    private readonly MLContext _mlContext;
    private PredictionEngine<ModelInput, ModelOutput> _predictionEngine;
    private ITransformer _trainedModel;
    private DataLoader _dataLoader;

    public Predictor(DataLoader dataLoader)
    {
        _mlContext = new(111);

        if(!File.Exists(_modelPath))
            throw new FileNotFoundException(_modelPath);

        using var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        _trainedModel = _mlContext.Model.Load(stream, out _);

        if (_trainedModel is null)
            throw new Exception("Failed to load model");

        _predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_trainedModel);
        _dataLoader = dataLoader;
    }

    public IEnumerable<ModelOutput> MakeTestDatasetPredictions()
    {
        var predictionData = _trainedModel.Transform(_dataLoader.TestSet);

        return _mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);
    }
}
