import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export async function getHealth() {
  const response = await client.get("/health");
  return response.data.data;
}

export async function getClasses() {
  const response = await client.get("/classes");
  return response.data.data;
}

export async function predictFrame(image, threshold, topK = 3) {
  const response = await client.post("/predict", {
    image,
    threshold,
    top_k: topK,
  });
  return response.data.data;
}

export function getApiError(exc) {
  const error = exc.response?.data?.error;
  return {
    code: error?.code ?? "request_failed",
    message: error?.message ?? "Prediction request failed.",
    details: error?.details ?? null,
    status: exc.response?.status ?? null,
  };
}
