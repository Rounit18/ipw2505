import { useCallback, useEffect, useState } from "react";

import { getHealth } from "../services/api.js";

export function useHealth() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const health = await getHealth();
      setData(health);
      setError(null);
    } catch (exc) {
      setError("Backend is unreachable.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;

    async function poll() {
      try {
        const health = await getHealth();
        if (active) {
          setData(health);
          setError(null);
        }
      } catch (exc) {
        if (active) {
          setError("Backend is unreachable.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    poll();
    const id = window.setInterval(poll, 5000);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, []);

  return { data, error, loading, retry: load };
}
