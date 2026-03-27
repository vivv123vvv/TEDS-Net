from evaluate_results import build_argument_parser, run_evaluation


def main():
    parser = build_argument_parser(description="重新导出 ACDC 评估可视化。")
    parsed_args = parser.parse_args()
    run_evaluation(args=parsed_args)


if __name__ == "__main__":
    main()
