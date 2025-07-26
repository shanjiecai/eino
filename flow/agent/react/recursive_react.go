/*
 * Copyright 2024 CloudWeGo Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package react

import (
	"context"
	"io"
	"strings"
	"sync"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/flow/agent"
	"github.com/cloudwego/eino/schema"
)

// RecursiveState 递归 React agent 的状态
type RecursiveState struct {
	Messages                 []*schema.Message
	ReturnDirectlyToolCallID string
	Round                    int
	LastAction               string // "chat", "tool", "start"
	ConsecutiveToolCalls     int
	ConsecutiveChats         int
	UserContext              map[string]interface{}
}

// RecursiveFlowController 递归流程控制器类型
type RecursiveFlowController func(ctx context.Context, state *RecursiveState, lastMessage *schema.Message) (nextAction string, err error)

// RecursiveAgentConfig 递归 React agent 的配置
type RecursiveAgentConfig struct {
	// ToolCallingModel is the chat model to be used for handling user messages with tool calling capability.
	ToolCallingModel model.ToolCallingChatModel

	// Deprecated: Use ToolCallingModel instead.
	Model model.ChatModel

	// ToolsConfig is the config for tools node.
	ToolsConfig compose.ToolsNodeConfig

	// MessageModifier.
	MessageModifier MessageModifier

	// MaxStep.
	MaxStep int `json:"max_step"`

	// Tools that will make agent return directly when the tool is called.
	ToolReturnDirectly map[string]struct{}

	// StreamToolCallChecker for streaming mode
	StreamToolCallChecker func(ctx context.Context, modelOutput *schema.StreamReader[*schema.Message]) (bool, error)

	// Graph and node names
	GraphName     string
	ModelNodeName string
	ToolsNodeName string

	// 递归流程控制器，支持任意的 tool-chat 组合
	// 返回值：nextAction 可以是 "chat", "tool", "end"
	RecursiveFlowController RecursiveFlowController

	// 最大递归深度，防止无限循环（0表示无限制）
	MaxRecursionDepth int

	// 允许连续工具调用的最大次数（0表示无限制）
	MaxConsecutiveTools int

	// 允许连续聊天的最大次数（0表示无限制）
	MaxConsecutiveChats int
}

// RecursiveAgent 递归 React agent
type RecursiveAgent struct {
	runnable         compose.Runnable[[]*schema.Message, *schema.Message]
	graph            *compose.Graph[[]*schema.Message, *schema.Message]
	graphAddNodeOpts []compose.GraphAddNodeOpt
}

var registerRecursiveStateOnce sync.Once

const (
	recursiveNodeKeyTools = "tools"
	recursiveNodeKeyModel = "chat"
	recursiveGraphName    = "RecursiveReActAgent"
	recursiveModelName    = "RecursiveChatModel"
	recursiveToolsName    = "RecursiveTools"
)

// SetRecursiveNextAction 设置递归 agent 的下一步行为
func SetRecursiveNextAction(ctx context.Context, nextAction string) error {
	return compose.ProcessState(ctx, func(ctx context.Context, s *RecursiveState) error {
		if s.UserContext == nil {
			s.UserContext = make(map[string]interface{})
		}
		s.UserContext["force_next_action"] = nextAction
		return nil
	})
}

// SetRecursiveUserContext 设置递归 agent 的用户上下文
func SetRecursiveUserContext(ctx context.Context, key string, value interface{}) error {
	return compose.ProcessState(ctx, func(ctx context.Context, s *RecursiveState) error {
		if s.UserContext == nil {
			s.UserContext = make(map[string]interface{})
		}
		s.UserContext[key] = value
		return nil
	})
}

// GetRecursiveUserContext 获取递归 agent 的用户上下文
func GetRecursiveUserContext(ctx context.Context, key string) (interface{}, error) {
	var value interface{}
	err := compose.ProcessState(ctx, func(ctx context.Context, s *RecursiveState) error {
		if s.UserContext != nil {
			value = s.UserContext[key]
		}
		return nil
	})
	return value, err
}

// GetRecursiveCurrentState 获取递归 agent 的当前状态
func GetRecursiveCurrentState(ctx context.Context) (*RecursiveState, error) {
	var currentState *RecursiveState
	err := compose.ProcessState(ctx, func(ctx context.Context, s *RecursiveState) error {
		currentState = &RecursiveState{
			Round:                    s.Round,
			LastAction:               s.LastAction,
			ConsecutiveToolCalls:     s.ConsecutiveToolCalls,
			ConsecutiveChats:         s.ConsecutiveChats,
			ReturnDirectlyToolCallID: s.ReturnDirectlyToolCallID,
		}
		if s.UserContext != nil {
			currentState.UserContext = make(map[string]interface{})
			for k, v := range s.UserContext {
				currentState.UserContext[k] = v
			}
		}
		return nil
	})
	return currentState, err
}

// DefaultRecursiveFlowController 默认的递归流程控制器
// 支持无限制的 tool-chat 任意组合
func DefaultRecursiveFlowController() RecursiveFlowController {
	return func(ctx context.Context, state *RecursiveState, lastMessage *schema.Message) (string, error) {
		// 检查强制设置的下一步行为
		if state.UserContext != nil {
			if forceAction, exists := state.UserContext["force_next_action"]; exists {
				if action, ok := forceAction.(string); ok && action != "" {
					// 清除强制设置
					state.UserContext["force_next_action"] = ""
					return action, nil
				}
			}
		}

		// 如果有工具调用，默认执行工具
		if lastMessage != nil && len(lastMessage.ToolCalls) > 0 {
			return "tool", nil
		}

		// 检查是否应该结束
		if state.UserContext != nil {
			if shouldEnd, exists := state.UserContext["should_end"]; exists {
				if end, ok := shouldEnd.(bool); ok && end {
					return "end", nil
				}
			}
		}

		// 默认情况：没有工具调用且有内容则结束，否则继续聊天
		if lastMessage != nil && len(lastMessage.Content) > 0 && len(lastMessage.ToolCalls) == 0 {
			return "end", nil
		}

		return "chat", nil
	}
}

// FlexibleRecursiveFlowController 灵活的递归流程控制器
// 支持复杂的条件判断和任意组合
func FlexibleRecursiveFlowController(
	maxDepth int,
	maxConsecutiveTools int,
	maxConsecutiveChats int,
) RecursiveFlowController {
	return func(ctx context.Context, state *RecursiveState, lastMessage *schema.Message) (string, error) {
		// 1. 检查强制设置
		if state.UserContext != nil {
			if forceAction, exists := state.UserContext["force_next_action"]; exists {
				if action, ok := forceAction.(string); ok && action != "" {
					state.UserContext["force_next_action"] = ""
					return action, nil
				}
			}
		}

		// 2. 检查最大递归深度
		if maxDepth > 0 && state.Round >= maxDepth {
			return "end", nil
		}

		// 3. 检查连续操作限制
		if maxConsecutiveTools > 0 && state.ConsecutiveToolCalls >= maxConsecutiveTools {
			// 强制切换到聊天
			return "chat", nil
		}

		if maxConsecutiveChats > 0 && state.ConsecutiveChats >= maxConsecutiveChats {
			// 如果有工具调用，优先执行工具，否则结束
			if lastMessage != nil && len(lastMessage.ToolCalls) > 0 {
				return "tool", nil
			}
			return "end", nil
		}

		// 4. 基于消息内容的智能决策
		if lastMessage != nil {
			// 有工具调用就执行工具
			if len(lastMessage.ToolCalls) > 0 {
				return "tool", nil
			}

			// 检查内容是否需要继续对话
			content := strings.ToLower(lastMessage.Content)
			needsContinuation := strings.Contains(content, "继续") ||
				strings.Contains(content, "再") ||
				strings.Contains(content, "还有") ||
				strings.Contains(content, "另外")

			if needsContinuation {
				return "chat", nil
			}

			// 检查是否应该调用工具
			needsTools := strings.Contains(content, "使用") ||
				strings.Contains(content, "调用") ||
				strings.Contains(content, "工具") ||
				strings.Contains(content, "执行")

			if needsTools {
				// 这里可以设置提示模型使用工具
				SetRecursiveUserContext(ctx, "suggest_tool_use", true)
				return "chat", nil
			}
		}

		// 5. 检查结束条件
		if state.UserContext != nil {
			if shouldEnd, exists := state.UserContext["should_end"]; exists {
				if end, ok := shouldEnd.(bool); ok && end {
					return "end", nil
				}
			}
		}

		// 6. 默认决策：有内容但无工具调用则结束
		if lastMessage != nil && len(lastMessage.Content) > 0 && len(lastMessage.ToolCalls) == 0 {
			return "end", nil
		}

		// 7. 其他情况继续聊天
		return "chat", nil
	}
}

// NewRecursiveAgent 创建递归 React agent
func NewRecursiveAgent(ctx context.Context, config *RecursiveAgentConfig) (_ *RecursiveAgent, err error) {
	var (
		chatModel       model.BaseChatModel
		toolsNode       *compose.ToolsNode
		toolInfos       []*schema.ToolInfo
		toolCallChecker = config.StreamToolCallChecker
		messageModifier = config.MessageModifier
	)

	// 注册递归状态类型
	registerRecursiveStateOnce.Do(func() {
		err = compose.RegisterSerializableType[RecursiveState]("_eino_recursive_react_state")
	})
	if err != nil {
		return nil, err
	}

	// 设置默认名称
	graphName := recursiveGraphName
	if config.GraphName != "" {
		graphName = config.GraphName
	}

	modelNodeName := recursiveModelName
	if config.ModelNodeName != "" {
		modelNodeName = config.ModelNodeName
	}

	toolsNodeName := recursiveToolsName
	if config.ToolsNodeName != "" {
		toolsNodeName = config.ToolsNodeName
	}

	// 设置默认流程控制器
	flowController := config.RecursiveFlowController
	if flowController == nil {
		flowController = DefaultRecursiveFlowController()
	}

	// 设置默认工具调用检查器
	if toolCallChecker == nil {
		toolCallChecker = firstChunkStreamToolCallChecker
	}

	// 准备工具信息
	if toolInfos, err = genToolInfos(ctx, config.ToolsConfig); err != nil {
		return nil, err
	}

	// 准备聊天模型
	if chatModel, err = agent.ChatModelWithTools(config.Model, config.ToolCallingModel, toolInfos); err != nil {
		return nil, err
	}

	// 准备工具节点
	if toolsNode, err = compose.NewToolNode(ctx, &config.ToolsConfig); err != nil {
		return nil, err
	}

	// 创建图
	graph := compose.NewGraph[[]*schema.Message, *schema.Message](
		compose.WithGenLocalState(func(ctx context.Context) *RecursiveState {
			return &RecursiveState{
				Messages:             make([]*schema.Message, 0, config.MaxStep+1),
				Round:                0,
				LastAction:           "start",
				ConsecutiveToolCalls: 0,
				ConsecutiveChats:     0,
				UserContext:          make(map[string]interface{}),
			}
		}))

	// 聊天模型预处理
	modelPreHandle := func(ctx context.Context, input []*schema.Message, state *RecursiveState) ([]*schema.Message, error) {
		// 更新状态
		state.Messages = append(state.Messages, input...)
		state.Round++

		// 更新连续操作计数
		if state.LastAction == "chat" {
			state.ConsecutiveChats++
			state.ConsecutiveToolCalls = 0
		} else {
			state.ConsecutiveChats = 1 // 重置为1，因为当前是chat
			state.ConsecutiveToolCalls = 0
		}
		state.LastAction = "chat"

		// 应用消息修饰器
		if messageModifier == nil {
			return state.Messages, nil
		}

		modifiedInput := make([]*schema.Message, len(state.Messages))
		copy(modifiedInput, state.Messages)
		return messageModifier(ctx, modifiedInput), nil
	}

	// 添加聊天模型节点
	if err = graph.AddChatModelNode(recursiveNodeKeyModel, chatModel,
		compose.WithStatePreHandler(modelPreHandle),
		compose.WithNodeName(modelNodeName)); err != nil {
		return nil, err
	}

	// 添加开始边
	if err = graph.AddEdge(compose.START, recursiveNodeKeyModel); err != nil {
		return nil, err
	}

	// 工具节点预处理
	toolsNodePreHandle := func(ctx context.Context, input *schema.Message, state *RecursiveState) (*schema.Message, error) {
		if input == nil {
			return state.Messages[len(state.Messages)-1], nil
		}

		state.Messages = append(state.Messages, input)
		state.ReturnDirectlyToolCallID = getReturnDirectlyToolCallID(input, config.ToolReturnDirectly)

		// 更新连续操作计数
		if state.LastAction == "tool" {
			state.ConsecutiveToolCalls++
			state.ConsecutiveChats = 0
		} else {
			state.ConsecutiveToolCalls = 1 // 重置为1，因为当前是tool
			state.ConsecutiveChats = 0
		}
		state.LastAction = "tool"

		return input, nil
	}

	// 添加工具节点
	if err = graph.AddToolsNode(recursiveNodeKeyTools, toolsNode,
		compose.WithStatePreHandler(toolsNodePreHandle),
		compose.WithNodeName(toolsNodeName)); err != nil {
		return nil, err
	}

	// 递归分支条件：从聊天模型出发的分支决策
	chatModelBranchCondition := func(ctx context.Context, sr *schema.StreamReader[*schema.Message]) (endNode string, err error) {
		// 收集流式消息
		var collectedMessage *schema.Message
		messages := make([]*schema.Message, 0)

		for {
			msg, err := sr.Recv()
			if err == io.EOF {
				break
			}
			if err != nil {
				return "", err
			}
			messages = append(messages, msg)
		}

		// 合并消息
		if len(messages) > 0 {
			collectedMessage, err = schema.ConcatMessages(messages)
			if err != nil {
				return "", err
			}
		}

		// 获取当前状态
		var currentState *RecursiveState
		err = compose.ProcessState[*RecursiveState](ctx, func(_ context.Context, state *RecursiveState) error {
			currentState = state
			return nil
		})
		if err != nil {
			return "", err
		}

		// 使用流程控制器决定下一步
		nextAction, err := flowController(ctx, currentState, collectedMessage)
		if err != nil {
			return "", err
		}

		switch nextAction {
		case "tool":
			return recursiveNodeKeyTools, nil
		case "chat":
			return recursiveNodeKeyModel, nil // 递归回到聊天模型
		case "end":
			return compose.END, nil
		default:
			return compose.END, nil
		}
	}

	// 添加聊天模型的分支
	if err = graph.AddBranch(recursiveNodeKeyModel,
		compose.NewStreamGraphBranch(chatModelBranchCondition,
			map[string]bool{
				recursiveNodeKeyTools: true,
				recursiveNodeKeyModel: true, // 允许递归回到自己
				compose.END:           true,
			})); err != nil {
		return nil, err
	}

	// 工具节点的分支条件：支持 tool->tool 和 tool->chat
	toolsBranchCondition := func(ctx context.Context, msgsStream *schema.StreamReader[[]*schema.Message]) (endNode string, err error) {
		msgsStream.Close()

		// 获取当前状态
		var currentState *RecursiveState
		err = compose.ProcessState[*RecursiveState](ctx, func(_ context.Context, state *RecursiveState) error {
			currentState = state
			return nil
		})
		if err != nil {
			return "", err
		}

		// 检查是否需要直接返回
		if len(currentState.ReturnDirectlyToolCallID) > 0 {
			return compose.END, nil
		}

		// 使用流程控制器决定下一步（传入最后一条消息）
		var lastMessage *schema.Message
		if len(currentState.Messages) > 0 {
			lastMessage = currentState.Messages[len(currentState.Messages)-1]
		}

		nextAction, err := flowController(ctx, currentState, lastMessage)
		if err != nil {
			return "", err
		}

		switch nextAction {
		case "tool":
			// 如果下一步还是工具，需要先回到聊天模型让它决定调用什么工具
			return recursiveNodeKeyModel, nil
		case "chat":
			return recursiveNodeKeyModel, nil
		case "end":
			return compose.END, nil
		default:
			return recursiveNodeKeyModel, nil // 默认回到聊天模型
		}
	}

	// 添加工具节点的分支
	if err = graph.AddBranch(recursiveNodeKeyTools,
		compose.NewStreamGraphBranch(toolsBranchCondition,
			map[string]bool{
				recursiveNodeKeyModel: true,
				compose.END:           true,
			})); err != nil {
		return nil, err
	}

	// 编译图
	compileOpts := []compose.GraphCompileOption{
		compose.WithMaxRunSteps(config.MaxStep),
		compose.WithNodeTriggerMode(compose.AnyPredecessor),
		compose.WithGraphName(graphName),
	}

	runnable, err := graph.Compile(ctx, compileOpts...)
	if err != nil {
		return nil, err
	}

	return &RecursiveAgent{
		runnable:         runnable,
		graph:            graph,
		graphAddNodeOpts: []compose.GraphAddNodeOpt{compose.WithGraphCompileOptions(compileOpts...)},
	}, nil
}

// Generate 生成响应
func (r *RecursiveAgent) Generate(ctx context.Context, input []*schema.Message, opts ...agent.AgentOption) (*schema.Message, error) {
	return r.runnable.Invoke(ctx, input, agent.GetComposeOptions(opts...)...)
}

// Stream 流式生成响应
func (r *RecursiveAgent) Stream(ctx context.Context, input []*schema.Message, opts ...agent.AgentOption) (output *schema.StreamReader[*schema.Message], err error) {
	return r.runnable.Stream(ctx, input, agent.GetComposeOptions(opts...)...)
}

// ExportGraph 导出图
func (r *RecursiveAgent) ExportGraph() (compose.AnyGraph, []compose.GraphAddNodeOpt) {
	return r.graph, r.graphAddNodeOpts
}
