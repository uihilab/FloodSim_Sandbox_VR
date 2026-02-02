// Fill out your copyright notice in the Description page of Project Settings.


#include "Floodmap.h"
#include "Components/SceneComponent.h"
#include "Json.h"
#include "Floodsim.h"
#include "Misc/Paths.h"
#include "Misc/FileHelper.h"

// Sets default values
AFloodmap::AFloodmap()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;
	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("SceneComponent"));
	ProceduralMeshComponent = CreateDefaultSubobject<UProceduralMeshComponent>(TEXT("ProceduralMeshComponent"));
	ProceduralMeshComponent->SetupAttachment(RootComponent);
}

// Called when the game starts or when spawned
void AFloodmap::BeginPlay()
{
	Super::BeginPlay();
	// MakeHTTPRequest();
	//ReadFile();

	/*FString path = FPaths::ProjectDir() / TEXT("iowa_city_Two.geojson");
	FString jsonString{};
	FFileHelper::LoadFileToString(jsonString, *path);

	UE_LOG(LogTemp, Display, TEXT("%s"), *jsonString);*/

	ReadFile();
}

void AFloodmap::ReadFile()
{
	UFloodsim* FloodInstance = Cast<UFloodsim>(GetGameInstance());

	if (FloodInstance)
	{
		ParseFile(FloodInstance->JsonString);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Could not cast game instance in floodplain class."));
	}
}

void AFloodmap::ParseFile(const FString& FileContent)
{
	TSharedPtr<FJsonObject> JsonObject{};
	TSharedRef<TJsonReader<>> JsonReader = TJsonReaderFactory<>::Create(FileContent);

	if (FJsonSerializer::Deserialize(JsonReader, JsonObject))			// Error occuring here.
	{
		if (JsonObject.IsValid())
		{
			UE_LOG(LogTemp, Display, TEXT("Loading Json valid"));

			const TArray<TSharedPtr<FJsonValue>>* Features;
			if (JsonObject->TryGetArrayField(TEXT("features"), Features))									// Possible site for optimization.
			{
				for (const TSharedPtr<FJsonValue>& Feature : *Features)
				{
					TSharedPtr<FJsonObject> Geometry = Feature->AsObject()->GetObjectField("geometry");

					const TArray<TSharedPtr<FJsonValue>>* CoordinatesArray;
					if (Geometry->TryGetArrayField(TEXT("coordinates"), CoordinatesArray))
					{
						for (const TSharedPtr<FJsonValue>& FirstLevelArray : *CoordinatesArray)
						{
							if (FirstLevelArray->Type == EJson::Array)
							{
								const TArray<TSharedPtr<FJsonValue>> VerticesArray = FirstLevelArray->AsArray();
								TArray<FVector> WorldVertices{};

								for (const TSharedPtr<FJsonValue>& VertexJson : VerticesArray)
								{
									if (VertexJson->Type == EJson::Array)
									{
										const TArray<TSharedPtr<FJsonValue>> Points = VertexJson->AsArray();

										double X = Points[0]->AsNumber();
										double Y = Points[1]->AsNumber();
										double Z = Points.Num() > 2 ? Points[2]->AsNumber() : 0.0;

										/*UE_LOG(LogTemp, Display, TEXT("Long: %f"), X);
										UE_LOG(LogTemp, Display, TEXT("Lat: %f"), Y);
										UE_LOG(LogTemp, Display, TEXT("Alt: %f"), Z);

										FString X = Points[0]->AsString();
										FString Y = Points[1]->AsString();
										FString Z = Points.Num() > 2 ? Points[2]->AsString() : FString("0");*/

										FVector WorldVertex = LatLongToWorldspace(X, Y, Z);
										WorldVertices.Add(WorldVertex);
									}
								}

								DrawFloodMap(WorldVertices);
							}
						}
					}
				}
			}
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("Json object is invalid."));
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Could not deserealize json object from json reader"));
	}
}

void AFloodmap::MakeHTTPRequest()
{
	UE_LOG(LogTemp, Display, TEXT("Making request..."));

	FHttpRequestRef Request = FHttpModule::Get().CreateRequest();
	Request->OnProcessRequestComplete().BindUObject(this, &AFloodmap::OnResponseReceived);
	Request->SetURL("http://127.0.0.1:5000/");
	Request->SetVerb("GET");
	Request->SetHeader("Content-Type", "application/json");
	Request->ProcessRequest();
}

void AFloodmap::OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool wasSuccessful)
{
	UE_LOG(LogTemp, Display, TEXT("Response received."))

		if (wasSuccessful && Response->GetResponseCode() == 200)
		{
			UE_LOG(LogTemp, Display, TEXT("Request successful!"))

			FString JsonContent = Response->GetContentAsString();

			UE_LOG(LogTemp, Display, TEXT("%s"), *JsonContent);

			TSharedPtr<FJsonObject> JsonObject{};
			TSharedRef<TJsonReader<>> JsonReader = TJsonReaderFactory<>::Create(Response->GetContentAsString());

			if (FJsonSerializer::Deserialize(JsonReader, JsonObject) && JsonObject.IsValid())
			{
				const TArray<TSharedPtr<FJsonValue>>* Features;
				if (JsonObject->TryGetArrayField(TEXT("features"), Features))									// Possible site for optimization.
				{
					for (const TSharedPtr<FJsonValue>& Feature : *Features)
					{
						TSharedPtr<FJsonObject> Geometry = Feature->AsObject()->GetObjectField("geometry");

						const TArray<TSharedPtr<FJsonValue>>* CoordinatesArray;
						if (Geometry->TryGetArrayField(TEXT("coordinates"), CoordinatesArray))
						{
							for (const TSharedPtr<FJsonValue>& FirstLevelArray : *CoordinatesArray)
							{
								if (FirstLevelArray->Type == EJson::Array)
								{
									const TArray<TSharedPtr<FJsonValue>> VerticesArray = FirstLevelArray->AsArray();
									TArray<FVector> WorldVertices{};

									for (const TSharedPtr<FJsonValue>& VertexJson : VerticesArray)
									{
										if (VertexJson->Type == EJson::Array)
										{
											const TArray<TSharedPtr<FJsonValue>> Points = VertexJson->AsArray();

											double X = Points[0]->AsNumber();
											double Y = Points[1]->AsNumber();
											double Z = Points.Num() > 2 ? Points[2]->AsNumber() : 0.0;

											/*UE_LOG(LogTemp, Display, TEXT("Long: %f"), X);
											UE_LOG(LogTemp, Display, TEXT("Lat: %f"), Y);
											UE_LOG(LogTemp, Display, TEXT("Alt: %f"), Z);

											FString X = Points[0]->AsString();
											FString Y = Points[1]->AsString();
											FString Z = Points.Num() > 2 ? Points[2]->AsString() : FString("0");*/

											FVector WorldVertex = LatLongToWorldspace(X, Y, Z);
											WorldVertices.Add(WorldVertex);
										}
									}

									DrawFloodMap(WorldVertices);
								}
							}
						}
					}
				}
			}
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("Request failed."));
		}
}

FVector AFloodmap::LatLongToWorldspace(double longitude, double latitude, double altitude)
{
	if (GeoReferencingSystem == nullptr)
	{
		UE_LOG(LogTemp, Error, TEXT("Georeferencing system is null."));
		return FVector::ZeroVector;
	}

	/*UE_LOG(LogTemp, Display, TEXT("Longitude: %f"), longitude);
	UE_LOG(LogTemp, Display, TEXT("Latitude: %f"), latitude);
	UE_LOG(LogTemp, Display, TEXT("Altitude: %f"), altitude);*/

	if (altitude == 0)
		altitude = 668.0;

	const FGeographicCoordinates Coordinates(longitude, latitude, altitude);
	FVector EngineLocation{};

	GeoReferencingSystem->GeographicToEngine(Coordinates, EngineLocation);
	return EngineLocation;
}

// Called every frame
void AFloodmap::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void AFloodmap::DrawFloodMap(const TArray<FVector>& Vertices)
{
	Async(EAsyncExecution::ThreadPool, [this, Vertices]()
		{
			TArray<int32> Triangles{};

			for (int32 i = 0; i < Vertices.Num() - 2; ++i)
			{
				Triangles.Add(0);
				Triangles.Add(i + 1);
				Triangles.Add(i + 2);
			}

			// Push mesh creation and updates to game thread.
			AsyncTask(ENamedThreads::GameThread, [this, Vertices, Triangles]()
				{
					ProceduralMeshComponent->CreateMeshSection_LinearColor(CurrentMeshSectionIndex, Vertices, Triangles,
						TArray<FVector>(), TArray<FVector2D>(), TArray<FLinearColor>(), TArray<FProcMeshTangent>(), true);

					ProceduralMeshComponent->SetMaterial(CurrentMeshSectionIndex, Material);
					DynamicMat = ProceduralMeshComponent->CreateDynamicMaterialInstance(CurrentMeshSectionIndex, Material);
					InstancedMaterials.Add(DynamicMat);

					MeshSectionIndices.Add(CurrentMeshSectionIndex);
					CurrentMeshSectionIndex++;
				});
		});
}

