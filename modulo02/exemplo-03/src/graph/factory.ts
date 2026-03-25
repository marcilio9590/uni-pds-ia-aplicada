import { config } from "../config.ts";
import { AppointmentService } from "../services/appointmentService.ts";
import { OpenRouterService } from "../services/openRouterService.ts";
import { buildAppointmentGraph } from "./graph.ts";

export function buildGraph() {
  const llmCLient = new OpenRouterService(config);
  const appointmentService = new AppointmentService();

  return buildAppointmentGraph(llmCLient, appointmentService);
}

export const graph = async () => {
  return buildGraph();
};
