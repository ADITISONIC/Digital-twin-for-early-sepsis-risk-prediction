export type Prediction = {
  patient_id: string;
  hospital_id: string;
  timestamp: number;
  probability: number;
  risk: string;
  alert: boolean;
  threshold: number;
  explanation?: {
    base_value: number;
    features: { feature: string; value: number | null; shap: number; direction: string }[];
  };
};

const request = async (url: string, options?: RequestInit) => {
  const r = await fetch(url, options);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
};

export const api = {
  health: () => request("/api/health"),
  patients: () => request("/api/patients"),
  patient: (id: string) => request(`/api/patients/${id}`),
  predict: (payload: any): Promise<Prediction> =>
    request("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
};
