import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import Webcam from "react-webcam";

function frameHasVisualSignal(context, width, height) {
  const { data } = context.getImageData(0, 0, width, height);
  let sum = 0;
  let sumSquares = 0;
  let saturatedPixels = 0;
  const pixels = width * height;

  for (let index = 0; index < data.length; index += 4) {
    const red = data[index];
    const green = data[index + 1];
    const blue = data[index + 2];
    const max = Math.max(red, green, blue);
    const min = Math.min(red, green, blue);
    const luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    sum += luma;
    sumSquares += luma * luma;
    if (max - min > 18) {
      saturatedPixels += 1;
    }
  }

  const mean = sum / pixels;
  const variance = Math.max(0, sumSquares / pixels - mean * mean);
  const stdDev = Math.sqrt(variance);
  const saturationRatio = saturatedPixels / pixels;

  return stdDev >= 9 || saturationRatio >= 0.12;
}

const VideoFeed = forwardRef(function VideoFeed({ disabled, onCameraStateChange }, ref) {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraState, setCameraState] = useState(disabled ? "disabled" : "requesting");

  function updateCameraState(nextState) {
    setCameraState(nextState);
    onCameraStateChange?.(nextState);
  }

  useEffect(() => {
    if (disabled) {
      updateCameraState("disabled");
    } else if (cameraState === "disabled") {
      updateCameraState("requesting");
    }
  }, [disabled]);

  useImperativeHandle(ref, () => ({
    captureFrame() {
      const video = webcamRef.current?.video;
      const canvas = canvasRef.current;

      if (disabled || !video || !canvas || video.readyState < 2) {
        return null;
      }

      const context = canvas.getContext("2d");
      const sourceWidth = video.videoWidth;
      const sourceHeight = video.videoHeight;

      if (!sourceWidth || !sourceHeight) {
        return null;
      }

      const cropSize = Math.min(sourceWidth * 0.48, sourceHeight * 0.62);
      const sourceX = (sourceWidth - cropSize) / 2;
      const sourceY = (sourceHeight - cropSize) / 2;

      context.drawImage(
        video,
        sourceX,
        sourceY,
        cropSize,
        cropSize,
        0,
        0,
        64,
        64,
      );

      if (!frameHasVisualSignal(context, 64, 64)) {
        return {
          state: "no_hand",
          message: "No hand detected inside the guide box.",
        };
      }

      return {
        state: "ready",
        image: canvas.toDataURL("image/png"),
      };
    },
  }));

  let overlay = null;
  if (disabled) {
    overlay = "Prediction disabled until backend health is ok.";
  } else if (cameraState === "requesting") {
    overlay = "Waiting for camera permission.";
  } else if (cameraState === "denied") {
    overlay = "Camera permission denied.";
  } else if (cameraState === "unavailable") {
    overlay = "Camera unavailable.";
  }

  return (
    <section className="video-panel">
      <div className="video-frame">
        {disabled ? null : (
          <>
            <Webcam
              ref={webcamRef}
              mirrored
              audio={false}
              screenshotFormat="image/png"
              videoConstraints={{ facingMode: "user" }}
              className="webcam"
              onUserMedia={() => updateCameraState("ready")}
              onUserMediaError={(error) => {
                const denied = ["NotAllowedError", "PermissionDeniedError"].includes(error.name);
                updateCameraState(denied ? "denied" : "unavailable");
              }}
            />
            <div className="guide-box" aria-hidden="true" />
          </>
        )}
        {overlay ? <div className="camera-disabled">{overlay}</div> : null}
      </div>
      <canvas ref={canvasRef} width="64" height="64" hidden />
    </section>
  );
});

export default VideoFeed;
