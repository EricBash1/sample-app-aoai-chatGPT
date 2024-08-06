import { useContext, useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { nord } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { FontIcon, Stack, Text } from '@fluentui/react'
import { useBoolean } from '@fluentui/react-hooks'
import DOMPurify from 'dompurify'
import remarkGfm from 'remark-gfm'
import supersub from 'remark-supersub'
import { AskResponse, Citation } from '../../api'
import { XSSAllowTags, XSSAllowAttributes } from '../../constants/sanatizeAllowables'
import { AppStateContext } from '../../state/AppProvider'
import { parseAnswer } from './AnswerParser'
import styles from './Answer.module.css'

interface Props {
    answer: AskResponse
    onCitationClicked: (citedDocument: Citation) => void
    onIntentsClicked: (intents: string[]) => void
}

export const Answer = ({ answer, onCitationClicked, onIntentsClicked }: Props) => {


    const [isRefAccordionOpen, { toggle: toggleIsRefAccordionOpen }] = useBoolean(false)
    const filePathTruncationLimit = 50

    const parsedAnswer = useMemo(() => parseAnswer(answer), [answer])
    const [chevronIsExpanded, setChevronIsExpanded] = useState(isRefAccordionOpen)

    const appStateContext = useContext(AppStateContext)

    const SANITIZE_ANSWER = appStateContext?.state.frontendSettings?.sanitize_answer

    const handleChevronClick = () => {
        setChevronIsExpanded(!chevronIsExpanded)
        toggleIsRefAccordionOpen()
    }

    useEffect(() => {
        setChevronIsExpanded(isRefAccordionOpen)
    }, [isRefAccordionOpen])



    const createCitationFilepath = (citation: Citation, index: number, truncate: boolean = false) => {
        let citationFilename = ''
        if (citation.filepath) {
            const filePath = citation.filepath.substring(citation.filepath.lastIndexOf('/') + 1)
            const part_i = citation.part_index ?? (citation.chunk_id ? parseInt(citation.chunk_id) + 1 : '')
            if (truncate && filePath.length > filePathTruncationLimit) {
                const citationLength = filePath.length
                citationFilename = `${filePath.substring(0, 20)}...${filePath.substring(citationLength - 20)} - Part ${part_i}`
            } else {
                citationFilename = `${filePath} - Part ${part_i}`
            }
        } else if (citation.filepath && citation.reindex_id) {
            citationFilename = `${citation.filepath} - Part ${citation.reindex_id}`
        } else {
            citationFilename = `Citation ${index}`
        }
        return citationFilename
    }


    const components = {
        code({ node, ...props }: { node: any;[key: string]: any }) {
            let language
            if (props.className) {
                const match = props.className.match(/language-(\w+)/)
                language = match ? match[1] : undefined
            }
            const codeString = node.children[0].value ?? ''
            return (
                <SyntaxHighlighter style={nord} language={language} PreTag="div" {...props}>
                    {codeString}
                </SyntaxHighlighter>
            )
        }
    }
    return (
        <>
            <Stack className={styles.answerContainer} tabIndex={0}>
                <Stack.Item>
                    <Stack horizontal grow>
                        <Stack.Item grow>
                            <ReactMarkdown
                                linkTarget="_blank"
                                remarkPlugins={[remarkGfm, supersub]}
                                children={
                                    SANITIZE_ANSWER
                                        ? DOMPurify.sanitize(parsedAnswer.markdownFormatText, { ALLOWED_TAGS: XSSAllowTags, ALLOWED_ATTR: XSSAllowAttributes })
                                        : parsedAnswer.markdownFormatText
                                }
                                className={styles.answerText}
                                components={components}
                            />
                        </Stack.Item>
                    </Stack>
                </Stack.Item>
                <Stack horizontal className={styles.answerFooter}>
                    {!!parsedAnswer.citations.length && (
                        <Stack.Item onKeyDown={e => (e.key === 'Enter' || e.key === ' ' ? toggleIsRefAccordionOpen() : null)}>
                            <Stack style={{ width: '100%' }}>
                                <Stack horizontal horizontalAlign="start" verticalAlign="center">
                                    <Text
                                        className={styles.accordionTitle}
                                        onClick={toggleIsRefAccordionOpen}
                                        aria-label="Open references"
                                        tabIndex={0}
                                        role="button">
                                        <span>
                                            {parsedAnswer.citations.length > 1
                                                ? parsedAnswer.citations.length + ' references'
                                                : '1 reference'}
                                        </span>
                                    </Text>
                                    <FontIcon
                                        className={styles.accordionIcon}
                                        onClick={handleChevronClick}
                                        iconName={chevronIsExpanded ? 'ChevronDown' : 'ChevronRight'}
                                    />
                                </Stack>
                            </Stack>
                        </Stack.Item>
                    )}
                    <Stack.Item className={styles.answerDisclaimerContainer}>
                        <span className={styles.answerDisclaimer}>AI-generated content may be incorrect</span>
                    </Stack.Item>
                    {!!answer.intents?.length && (
                        <Stack.Item onKeyDown={e => (e.key === 'Enter' || e.key === ' ' ? toggleIsRefAccordionOpen() : null)}>
                            <Stack style={{ width: '100%' }}>
                                <Stack horizontal horizontalAlign="start" verticalAlign="center">
                                    <Text
                                        className={styles.accordionTitle}
                                        onClick={() => onIntentsClicked(answer.intents)}
                                        aria-label="Open Intents"
                                        tabIndex={0}
                                        role="button">
                                        <span>
                                            Show Intents
                                        </span>
                                    </Text>
                                    <FontIcon
                                        className={styles.accordionIcon}
                                        onClick={handleChevronClick}
                                        iconName={'ChevronRight'}
                                    />
                                </Stack>
                            </Stack>
                        </Stack.Item>
                    )}
                </Stack>
                {chevronIsExpanded && (
                    <div className={styles.citationWrapper}>
                        {parsedAnswer.citations.map((citation, idx) => {
                            return (
                                <span
                                    title={createCitationFilepath(citation, ++idx)}
                                    tabIndex={0}
                                    role="link"
                                    key={idx}
                                    onClick={() => onCitationClicked(citation)}
                                    onKeyDown={e => (e.key === 'Enter' || e.key === ' ' ? onCitationClicked(citation) : null)}
                                    className={styles.citationContainer}
                                    aria-label={createCitationFilepath(citation, idx)}>
                                    <div className={styles.citation}>{idx}</div>
                                    {createCitationFilepath(citation, idx, true)}
                                </span>
                            )
                        })}
                    </div>
                )}
            </Stack>

        </>
    )
}