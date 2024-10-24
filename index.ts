// agent.ts
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatGroq } from "@langchain/groq";
import { ChatOpenAI } from "@langchain/openai";
import { MemorySaver } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { AskNewsSDK } from '@emergentmethods/asknews-typescript-sdk'
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { DynamicStructuredTool } from "@langchain/core/tools";
import * as dotenv from 'dotenv';
import { SqlToolkit } from "langchain/agents/toolkits/sql";
import { DataSource } from "typeorm";
import { SqlDatabase } from "langchain/sql_db";
import { END, Annotation } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";
import { JsonOutputToolsParser } from "@langchain/core/output_parsers/openai_tools";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { START, StateGraph } from "@langchain/langgraph";
import { SystemMessage } from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";

dotenv.config();

// IMPORTANT - Add your API keys here. Be careful not to publish them.
const GROQ_API_KEY=process.env.GROQ_API_KEY;
const TAVILY_API_KEY=process.env.TAVILY_API_KEY;
const LANGCHAIN_API_KEY=process.env.LANGCHAIN_API_KEY;
const LANGCHAIN_CALLBACKS_BACKGROUND=process.env.LANGCHAIN_CALLBACKS_BACKGROUND;
const LANGCHAIN_TRACING_V2=process.env.LANGCHAIN_TRACING_V2;
const LANGCHAIN_PROJECT=process.env.LANGCHAIN_PROJECT;

//const agentModel = new ChatGroq({ model: "llama-3.1-70b-Versatile", temperature: 0 });
const agentModel = new ChatOpenAI({ model: "gpt-4o", temperature: 0 });

const datasource = new DataSource({
  type: "sqlite",
  database: "c:/sqlite3/Chinook.db", // Replace with the link to your database
});

const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});

const ask = new AskNewsSDK({
  clientId: process.env.ASKNEWS_CLIENT_ID,
  clientSecret: process.env.ASKNEWS_CLIENT_SECRET,
  scopes: ['news'],
});

async function getNews(query) {
  const response = await ask.news.searchNews({
        query: query, // your keyword query
        nArticles: 5, // control the number of articles to include in the context
        returnType: 'dicts', // you can also ask for "dicts" if you want more information
        method: 'kw', // use "nl" for natural language for your search, or "kw" for keyword search
      });
      return JSON.stringify(response);
}

const toolkit = new SqlToolkit(db, agentModel);
const sqlTools = toolkit.getTools();

// Define the tools for the agent to use
const agentTools = [ 
    new DynamicStructuredTool({
      name: "AskNews",
      description: "Get current news and weather",
      schema: z.object({
        query: z.string().describe('Search Query'),
      }),
      func: async ({query}) => {
        return getNews(query)
      }
    }),
];

// Initialize memory to persist state between graph runs
const agentCheckpointer = new MemorySaver();
const agent = createReactAgent({
  llm: agentModel,
  tools: agentTools,
  checkpointSaver: agentCheckpointer,
});

// This defines the object that is passed between each node
// in the graph. We will create different nodes for each agent and tool
const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
    default: () => [],
  }),
  // The agent node that last performed work
  next: Annotation<string>({
    reducer: (x, y) => y ?? x ?? END,
    default: () => END,
  }),
});

const members = ["researcher", "chart_generator"] as const;

const systemPrompt =
  "You are a supervisor tasked with managing a conversation between the" +
  " following workers: {members}. Given the following user request," +
  " respond with the worker to act next. Each worker will perform a" +
  " task and respond with their results and status. When finished," +
  " respond with FINISH.";
const options = [END, ...members];

// Define the routing function
const routingTool = {
  name: "route",
  description: "Select the next role.",
  schema: z.object({
    next: z.enum([END, ...members]),
  }),
}



const prompt = ChatPromptTemplate.fromMessages([
  ["system", systemPrompt],
  new MessagesPlaceholder("messages"),
  [
    "system",
    "Given the conversation above, who should act next?" +
    " Or should we FINISH? Select one of: {options}",
  ],
]);

const formattedPrompt = await prompt.partial({
  options: options.join(", "),
  members: members.join(", "),
});

const llm = new ChatOpenAI({
  modelName: "gpt-4o",
  temperature: 0,
});

const supervisorChain = formattedPrompt
  .pipe(llm.bindTools(
    [routingTool],
    {
      tool_choice: "route",
    },
  ))
  .pipe(new JsonOutputToolsParser())
  // select the first one
  .pipe((x) => (x[0].args));

  await supervisorChain.invoke({
    messages: [
      new HumanMessage({
        content: "write a report on birds.",
      }),
    ],
  });

// Recall llm was defined as ChatOpenAI above
// It could be any other language model
const researcherAgent = createReactAgent({
  llm,
  tools: agentTools,
  messageModifier: new SystemMessage("You are a web researcher. You may use the Tavily search engine to search the web for" +
    " important information, so the Chart Generator in your team can make useful plots.")
})

const researcherNode = async (
  state: typeof AgentState.State,
  config?: RunnableConfig,
) => {
  const result = await researcherAgent.invoke(state, config);
  const lastMessage = result.messages[result.messages.length - 1];
  return {
    messages: [
      new HumanMessage({ content: lastMessage.content, name: "Researcher" }),
    ],
  };
};

const chartGenAgent = createReactAgent({
  llm,
  tools: toolkit,
  messageModifier: new SystemMessage("You excel at generating bar charts. Use the researcher's information to generate the charts.")
})

const chartGenNode = async (
  state: typeof AgentState.State,
  config?: RunnableConfig,
) => {
  const result = await chartGenAgent.invoke(state, config);
  const lastMessage = result.messages[result.messages.length - 1];
  return {
    messages: [
      new HumanMessage({ content: lastMessage.content, name: "ChartGenerator" }),
    ],
  };
};

// 1. Create the graph
const workflow = new StateGraph(AgentState)
  // 2. Add the nodes; these will do the work
  .addNode("researcher", researcherNode)
  .addNode("chart_generator", chartGenNode)
  .addNode("supervisor", supervisorChain);
// 3. Define the edges. We will define both regular and conditional ones
// After a worker completes, report to supervisor
members.forEach((member) => {
  workflow.addEdge(member, "supervisor");
});

workflow.addConditionalEdges(
  "supervisor",
  (x: typeof AgentState.State) => x.next,
);

workflow.addEdge(START, "supervisor");

const graph = workflow.compile();

// Now it's time to use!
const agentFinalState = await agent.invoke(
  { messages: [new HumanMessage("what is the five day forecast for Canisteo NY")] },
  { configurable: { thread_id: "42" } },
);

console.log(
  agentFinalState.messages[agentFinalState.messages.length - 1].content,
);

const agentNextState = await agent.invoke(
  { messages: [new HumanMessage("Can you list 10 artists from my database?")] },
  { configurable: { thread_id: "42" } },
);

console.log(
  agentNextState.messages[agentNextState.messages.length - 1].content,
);