// Fill out your copyright notice in the Description page of Project Settings.


#include "DataAnalysis.h"
#include "Engine/CanvasRenderTarget2D.h"
#include "ImageUtils.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/Base64.h"
#include "HttpModule.h"
#include "Json.h"
#include "JsonUtilities.h"
#include "Dom/JsonObject.h"

// Sets default values
ADataAnalysis::ADataAnalysis()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	RootComponent = CreateDefaultSubobject<USceneComponent>(TEXT("SceneComponent"));

	Camera = CreateDefaultSubobject<UCameraComponent>(TEXT("Camera"));
	Camera->SetupAttachment(RootComponent);

	SceneCaptureComponent = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("SceneCaptureCoponent2D"));
	SceneCaptureComponent->SetupAttachment(Camera);

	RenderTarget = CreateDefaultSubobject<UTextureRenderTarget2D>(TEXT("RenderTarget2D"));
	RenderTarget->InitCustomFormat(1920, 1080, PF_B8G8R8A8, false);

	SceneCaptureComponent->TextureTarget = RenderTarget;
	SceneCaptureComponent->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
}

// Called when the game starts or when spawned
void ADataAnalysis::BeginPlay()
{
	Super::BeginPlay();
}

// Called every frame
void ADataAnalysis::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void ADataAnalysis::TakeScreenshot(const FString FileName)
{
	if (!RenderTarget || !SceneCaptureComponent)
	{
		UE_LOG(LogTemp, Error, TEXT("Scene Capture or Render Target component is null. Aborting."))
			return;
	}

	FString SaveDirectory = FPaths::LaunchDir() / TEXT("Screenshots");

	if (!FPaths::DirectoryExists(*SaveDirectory))
	{
		IFileManager::Get().MakeDirectory(*SaveDirectory);
	}

	SceneCaptureComponent->CaptureScene();

	// Get render target resource and read pixels to bitmap.

	FTextureRenderTargetResource* RenderTargetResource = RenderTarget->GameThread_GetRenderTargetResource();
	TArray<FColor> Bitmap{};
	RenderTargetResource->ReadPixels(Bitmap);

	// UE_LOG(LogTemp, Display, TEXT("Number of pixels: %d"), Bitmap.Num());

	FString ImagePath = SaveDirectory / FileName;

	int32 XSize = RenderTarget->SizeX;
	int32 YSize = RenderTarget->SizeY;
	FImageUtils::CompressImageArray(XSize, YSize, Bitmap, CompressedMap);

	//UE_LOG(LogTemp, Display, TEXT("Size value X: %d"), XSize);
	//UE_LOG(LogTemp, Display, TEXT("Size value Y: %d"), YSize);
	//UE_LOG(LogTemp, Display, TEXT("Compressed map size: %d"), CompressedMap.Num());

	// Convert captured photo to UI Texture.

	Screenshot = UTexture2D::CreateTransient(XSize, YSize, PF_B8G8R8A8);

	if (!Screenshot)
		UE_LOG(LogTemp, Warning, TEXT("Failed to create Texture2D from screenshot"));

	void* TextureData = Screenshot->PlatformData->Mips[0].BulkData.Lock(LOCK_READ_WRITE);
	FMemory::Memcpy(TextureData, Bitmap.GetData(), Bitmap.Num() * sizeof(FColor));
	Screenshot->PlatformData->Mips[0].BulkData.Unlock();
	Screenshot->UpdateResource();

	// Save screenshot image to disk.

	if (FFileHelper::SaveArrayToFile(CompressedMap, *ImagePath))
	{
		UE_LOG(LogTemp, Display, TEXT("Image saved succesfully."));
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to save image."));
	}
}

void ADataAnalysis::AnalyzeScreenshot(const FString FileName)
{
	FString EncodedString = FBase64::Encode(CompressedMap);
	// UE_LOG(LogTemp, Display, TEXT("Encoded String: %s"), *EncodedString);
	MakeHTTPRequest(EncodedString);
}

void ADataAnalysis::MakeHTTPRequest(const FString& Base64String)
{
	TSharedRef<IHttpRequest> Request = FHttpModule::Get().CreateRequest();
	Request->OnProcessRequestComplete().BindUObject(this, &ADataAnalysis::OnResponseReceived);
	Request->SetURL("https://api.openai.com/v1/chat/completions");
	Request->SetVerb("POST");
	Request->SetHeader("Content-Type", "application/json");
	// TODO: Replace with your own OpenAI API key
	Request->SetHeader("Authorization", "Bearer YOUR_OPENAI_API_KEY_HERE");

	// Create json objects.

	TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject());
	JsonObject->SetStringField("model", "gpt-4o-mini");
	TSharedPtr<FJsonObject> MessageObject = MakeShareable(new FJsonObject());

	TArray<TSharedPtr<FJsonValue>> MessageArray{};
	TArray<TSharedPtr<FJsonValue>> ContentArray{};

	// Text Object.

	TSharedPtr<FJsonObject> TextObject = MakeShareable(new FJsonObject());
	TextObject->SetStringField("type", "text");
	TextObject->SetStringField("text", "Here is an image of a simulated flood in downtown Iowa City. Please inspect the image and recommend flood mitigation strategies that are specific to the presented scenario.");
	ContentArray.Add(MakeShareable(new FJsonValueObject(TextObject)));

	// Image object.

	TSharedPtr<FJsonObject> ImageObject = MakeShareable(new FJsonObject());
	ImageObject->SetStringField("url", FString::Printf(TEXT("data:image/png;base64,%s"), *Base64String)); // Updated MIME type
	TSharedPtr<FJsonObject> ImageUrlObject = MakeShareable(new FJsonObject());
	ImageUrlObject->SetStringField("type", "image_url"); // Ensure type is set correctly
	ImageUrlObject->SetObjectField("image_url", ImageObject);
	ContentArray.Add(MakeShareable(new FJsonValueObject(ImageUrlObject)));

	// Set messages.

	MessageObject->SetStringField("role", "user");
	MessageObject->SetArrayField("content", ContentArray);
	MessageArray.Add(MakeShareable(new FJsonValueObject(MessageObject)));
	JsonObject->SetArrayField("messages", MessageArray);
	JsonObject->SetNumberField("max_tokens", 300);

	// Serialize to json string.

	FString JsonOutput{};
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonOutput);
	FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer);

	// Submit.

	Request->SetContentAsString(JsonOutput);
	Request->ProcessRequest();
}

void ADataAnalysis::OnResponseReceived(FHttpRequestPtr Request, FHttpResponsePtr Response, bool wasSuccesful)
{
	if (wasSuccesful && Response.IsValid())
	{
		FString ResponseBody = Response->GetContentAsString();
		UE_LOG(LogTemp, Display, TEXT("%s"), *ResponseBody);

		TSharedPtr<FJsonObject> JsonObject{};
		TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseBody);

		if (FJsonSerializer::Deserialize(Reader, JsonObject) && JsonObject.IsValid())
		{
			// Access 'choices' array.
			TArray<TSharedPtr<FJsonValue>> Choices = JsonObject->GetArrayField("choices");

			if (Choices.Num() > 0)
			{
				TSharedPtr<FJsonObject> FirstChoice = Choices[0]->AsObject();

				if (FirstChoice.IsValid())
				{
					// Access message object.
					TSharedPtr<FJsonObject> Message = FirstChoice->GetObjectField("message");

					if (Message.IsValid())
					{
						// Get content from message.

						GPTResponse = Message->GetStringField("content");
						UE_LOG(LogTemp, Display, TEXT("Message Content: %s"), *GPTResponse);
					}
				}
			}
		}
		else
		{
			UE_LOG(LogTemp, Error, TEXT("Failed to deserialize Json response."))
		}
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Request unsuccessful"));
	}
}