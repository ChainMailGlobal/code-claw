import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const VOICE_CLAW_URL = Deno.env.get("VOICE_CLAW_URL") ?? "";
const VOICE_CLAW_SECRET = Deno.env.get("VOICE_CLAW_SECRET") ?? "";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface IncomingRequest {
  session_id: string;
  text: string;
  image_base64?: string;
  active_file?: string;
  stream?: boolean;  // true = SSE streaming, false = single response
}

function errorResponse(message: string, status: number): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json", ...corsHeaders },
  });
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (!VOICE_CLAW_URL) return errorResponse("VOICE_CLAW_URL not configured", 500);

  let body: IncomingRequest;
  try {
    body = await req.json();
  } catch {
    return errorResponse("Invalid JSON body", 400);
  }

  if (!body.text) return errorResponse("text is required", 400);
  if (!body.session_id) return errorResponse("session_id is required", 400);

  const authHeader = { Authorization: `Bearer ${VOICE_CLAW_SECRET}` };
  const payload: Record<string, unknown> = {
    session_id: body.session_id,
    instruction: body.text,
  };
  if (body.image_base64) payload.image_base64 = body.image_base64;
  if (body.active_file) payload.active_file = body.active_file;

  // Streaming mode — pipe SSE chunks directly to Spectacles Lens
  if (body.stream !== false) {
    const upstream = await fetch(`${VOICE_CLAW_URL}/execute/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader },
      body: JSON.stringify(payload),
    });

    if (!upstream.ok) {
      return errorResponse(`Stream failed: ${upstream.status}`, 502);
    }

    // Pass SSE stream straight through to the Lens
    return new Response(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        ...corsHeaders,
      },
    });
  }

  // Non-streaming fallback (iPhone / PC backdoor)
  const execResp = await fetch(`${VOICE_CLAW_URL}/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeader },
    body: JSON.stringify(payload),
  });

  if (!execResp.ok) return errorResponse(`Execute failed: ${execResp.status}`, 502);

  const result = await execResp.json();
  return new Response(JSON.stringify(result), {
    status: 200,
    headers: { "Content-Type": "application/json", ...corsHeaders },
  });
});
